"""Tests for scripts/measure_latency.py (T12).

``scripts/`` sits outside the ``triagem`` package (it is a CLI benchmark
tool, not library code -- see the T12 backlog entry), so it is loaded here
by file path instead of a normal package import. No network and no Docker
are used: the HTTP target's ``requests`` calls are monkeypatched, and the
in-process target trains a tiny real pipeline (same pattern as
``tests/serving/test_predictor.py``) on the deterministic
``tests/fixtures/laudos_sample.csv`` fixture instead of the full dataset --
the full ``data/processed/laudos.csv`` is gitignored and may not exist in a
fresh checkout (e.g. CI, before ``triagem.data.build`` has run).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import pytest

from triagem.data.loaders import VALID_LABELS, load_dataset
from triagem.models.pipeline import build_pipeline
from triagem.serving.predictor import Predictor
from triagem.training.train import LABEL_ENCODER_FILENAME, METRICS_FILENAME, MODEL_FILENAME

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "laudos_sample.csv"
SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "measure_latency.py"
TRAINED_AT = "2024-01-01T00:00:00+00:00"


def _load_measure_latency_module() -> ModuleType:
    """Import scripts/measure_latency.py by path (it lives outside the package)."""
    spec = importlib.util.spec_from_file_location("measure_latency", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ml = _load_measure_latency_module()


@pytest.fixture(autouse=True)
def _clear_predictor_cache() -> Iterator[None]:
    """Every test gets an empty Predictor cache, and leaves none behind."""
    Predictor.clear_cache()
    yield
    Predictor.clear_cache()


@pytest.fixture(scope="module")
def _fitted_pipeline() -> object:
    """Fit the real T6 pipeline on the small fixture once per test module."""
    df = load_dataset(FIXTURE_PATH)
    pipeline = build_pipeline(seed=42)
    pipeline.fit(df["text"].tolist(), df["label"].tolist())
    return pipeline


@pytest.fixture
def sklearn_models_dir(tmp_path: Path, _fitted_pipeline: object) -> Path:
    """A models/ dir with a real, in-memory-trained sklearn artifact + metadata."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(_fitted_pipeline, models_dir / MODEL_FILENAME)
    (models_dir / METRICS_FILENAME).write_text(
        json.dumps({"trained_at": TRAINED_AT, "macro_f1": 0.6}), encoding="utf-8"
    )
    (models_dir / LABEL_ENCODER_FILENAME).write_text(
        json.dumps({"classes": sorted(VALID_LABELS)}), encoding="utf-8"
    )
    return models_dir


def _counting_target(name: str = "fake") -> tuple[Any, list[str]]:
    """A Target that records every text it was called with, instantly."""
    calls: list[str] = []

    def _call(text: str) -> None:
        calls.append(text)

    return ml.Target(name=name, call=_call), calls


class TestSampleTexts:
    """sample_texts() -- deterministic, seeded input sampling."""

    def test_same_seed_yields_identical_texts(self) -> None:
        first = ml.sample_texts(30, seed=42, data_path=FIXTURE_PATH)
        second = ml.sample_texts(30, seed=42, data_path=FIXTURE_PATH)
        assert first == second

    def test_different_seed_yields_different_texts(self) -> None:
        first = ml.sample_texts(30, seed=1, data_path=FIXTURE_PATH)
        second = ml.sample_texts(30, seed=2, data_path=FIXTURE_PATH)
        assert first != second

    def test_returns_exactly_n_texts_even_beyond_pool_size(self) -> None:
        df = load_dataset(FIXTURE_PATH)
        texts = ml.sample_texts(len(df) * 3, seed=42, data_path=FIXTURE_PATH)
        assert len(texts) == len(df) * 3
        assert all(isinstance(t, str) and t for t in texts)


class TestMeasureLatency:
    """measure_latency() -- warm-up discard, percentile/throughput math."""

    def test_calls_target_n_plus_warmup_times(self) -> None:
        target, calls = _counting_target()
        ml.measure_latency(target, n=15, warmup=5, data_path=FIXTURE_PATH)
        assert len(calls) == 20

    def test_result_echoes_n_warmup_seed(self) -> None:
        target, _ = _counting_target()
        result = ml.measure_latency(target, n=10, warmup=3, seed=7, data_path=FIXTURE_PATH)
        assert result.n == 10
        assert result.warmup == 3
        assert result.seed == 7
        assert result.target == "fake"

    def test_percentiles_are_ordered_and_within_bounds(self) -> None:
        target, _ = _counting_target()
        result = ml.measure_latency(target, n=25, warmup=5, data_path=FIXTURE_PATH)
        assert result.min_ms <= result.p50_ms <= result.p90_ms <= result.p95_ms
        assert result.p95_ms <= result.p99_ms <= result.max_ms
        assert result.mean_ms >= 0.0
        assert result.stdev_ms >= 0.0
        assert result.throughput_rps > 0.0

    def test_warmup_calls_are_not_in_the_measured_count(self) -> None:
        # n=1 with warmup=0 must time exactly one call and never touch warm-up.
        target, calls = _counting_target()
        result = ml.measure_latency(target, n=1, warmup=0, data_path=FIXTURE_PATH)
        assert len(calls) == 1
        assert result.n == 1
        assert result.warmup == 0

    def test_n_must_be_positive(self) -> None:
        target, _ = _counting_target()
        with pytest.raises(ValueError, match="n must be positive"):
            ml.measure_latency(target, n=0, warmup=0, data_path=FIXTURE_PATH)

    def test_warmup_must_be_non_negative(self) -> None:
        target, _ = _counting_target()
        with pytest.raises(ValueError, match="warmup must be >= 0"):
            ml.measure_latency(target, n=5, warmup=-1, data_path=FIXTURE_PATH)


class TestInProcessSklearnTarget:
    """in_process_sklearn_target() -- real Predictor.load(backend='sklearn')."""

    def test_call_does_not_raise_and_predictor_is_reused(self, sklearn_models_dir: Path) -> None:
        target = ml.in_process_sklearn_target(models_dir=sklearn_models_dir)
        assert target.name == "in_process_sklearn"
        target.call("Patient presents with acute severe chest pain.")  # must not raise

    def test_measure_latency_end_to_end_in_process(self, sklearn_models_dir: Path) -> None:
        target = ml.in_process_sklearn_target(models_dir=sklearn_models_dir)
        result = ml.measure_latency(target, n=12, warmup=3, seed=42, data_path=FIXTURE_PATH)
        assert result.target == "in_process_sklearn"
        assert result.n == 12
        assert result.p50_ms >= 0.0


class TestHttpTarget:
    """http_target() / http_target_reachable() -- requests is always mocked."""

    def test_reachable_true_when_health_returns_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import requests

        class _Resp:
            status_code = 200

        monkeypatch.setattr(requests, "get", lambda url, timeout: _Resp())
        assert ml.http_target_reachable("http://127.0.0.1:8000") is True

    def test_reachable_false_when_status_is_not_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import requests

        class _Resp:
            status_code = 503

        monkeypatch.setattr(requests, "get", lambda url, timeout: _Resp())
        assert ml.http_target_reachable("http://127.0.0.1:8000") is False

    def test_reachable_false_when_connection_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import requests

        def _raise(url: str, timeout: float) -> None:
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(requests, "get", _raise)
        assert ml.http_target_reachable("http://127.0.0.1:8000") is False

    def test_call_posts_to_predict_and_checks_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import requests

        posted: dict[str, Any] = {}

        class _Resp:
            def raise_for_status(self) -> None:
                posted["raised_for_status_checked"] = True

        def _fake_post(self: Any, url: str, json: dict[str, Any], timeout: float) -> _Resp:
            posted["url"] = url
            posted["json"] = json
            return _Resp()

        monkeypatch.setattr(requests.Session, "post", _fake_post)
        target = ml.http_target("http://127.0.0.1:8000")
        assert target.name == "http_container"
        target.call("some report text")

        assert posted["url"] == "http://127.0.0.1:8000/predict"
        assert posted["json"] == {"text": "some report text"}
        assert posted["raised_for_status_checked"] is True


class TestMain:
    """main() -- writes docs/benchmarks/baseline.json + docs/latency_baseline.md."""

    def test_skip_http_writes_json_and_markdown_with_one_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sklearn_models_dir: Path
    ) -> None:
        json_out = tmp_path / "baseline.json"
        md_out = tmp_path / "latency_baseline.md"

        # main() resolves the dataset via Settings (no --data-path flag), so
        # point Settings.data_path at the fixture for this process only.
        from triagem.config import get_settings

        monkeypatch.setenv("TRIAGEM_DATA_PATH", str(FIXTURE_PATH))
        get_settings.cache_clear()

        exit_code = ml.main(
            [
                "--n",
                "10",
                "--warmup",
                "2",
                "--skip-http",
                "--models-dir",
                str(sklearn_models_dir),
                "--json-out",
                str(json_out),
                "--md-out",
                str(md_out),
            ]
        )

        get_settings.cache_clear()

        assert exit_code == 0
        assert json_out.exists()
        assert md_out.exists()

        payload = json.loads(json_out.read_text(encoding="utf-8"))
        assert payload["http_target_skipped_reason"] == "--skip-http passed"
        assert len(payload["results"]) == 1
        assert payload["results"][0]["target"] == "in_process_sklearn"
        assert payload["results"][0]["n"] == 10
        assert payload["results"][0]["warmup"] == 2
        assert "machine" in payload and "platform" in payload["machine"]

        markdown = md_out.read_text(encoding="utf-8")
        assert "# Baseline de latencia local em container" in markdown
        assert "in_process_sklearn" in markdown
        assert "nao medido" in markdown
