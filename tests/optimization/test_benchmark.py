"""Tests for triagem.optimization.benchmark (T21).

Fits the real T6 pipeline on the small, deterministic
``tests/fixtures/laudos_sample.csv`` fixture, exports it to ONNX (T20's
``export_to_onnx``), and exercises :func:`benchmark` through the real,
public :class:`~triagem.serving.predictor.Predictor` API for both backends
-- never a raw sklearn/onnxruntime call bypassing it. Equivalence-specific
behavior (the "core" gate this task must report honestly) has its own
dedicated coverage in ``tests/optimization/test_equivalence.py``; this file
covers the latency-measurement half plus the CLI/report-writing glue.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import joblib
import pytest
from sklearn.pipeline import Pipeline

from triagem.data.loaders import load_dataset
from triagem.models.pipeline import build_pipeline
from triagem.optimization import benchmark as bm
from triagem.optimization.onnx_export import export_to_onnx
from triagem.serving.predictor import Predictor
from triagem.training.train import MODEL_FILENAME

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "laudos_sample.csv"


@pytest.fixture(scope="module")
def fitted_pipeline() -> Pipeline:
    """Fit the real T6 pipeline on the small fixture once per test module."""
    df = load_dataset(FIXTURE_PATH)
    pipeline = build_pipeline(seed=42)
    pipeline.fit(df["text"].tolist(), df["label"].tolist())
    return pipeline


@pytest.fixture
def sklearn_only_models_dir(tmp_path: Path, fitted_pipeline: Pipeline) -> Path:
    """A models/ dir with only the sklearn artifact (no model.onnx)."""
    models_dir = tmp_path / "sklearn_only"
    models_dir.mkdir()
    joblib.dump(fitted_pipeline, models_dir / MODEL_FILENAME)
    return models_dir


@pytest.fixture
def both_backends_models_dir(sklearn_only_models_dir: Path) -> Path:
    """A models/ dir with both model.joblib and a real, exported model.onnx."""
    export_to_onnx(
        sklearn_only_models_dir / MODEL_FILENAME,
        sklearn_only_models_dir / "model.onnx",
    )
    return sklearn_only_models_dir


@pytest.fixture(autouse=True)
def _clear_predictor_cache() -> Iterator[None]:
    """Every test gets an empty Predictor cache, and leaves none behind."""
    Predictor.clear_cache()
    yield
    Predictor.clear_cache()


class TestSampleWithReplacement:
    """_sample_with_replacement() -- deterministic, seeded sampling helper."""

    def test_same_seed_yields_identical_samples(self) -> None:
        pool = ["a", "b", "c", "d"]
        first = bm._sample_with_replacement(pool, 10, seed=42)
        second = bm._sample_with_replacement(pool, 10, seed=42)
        assert first == second

    def test_different_seed_yields_different_samples(self) -> None:
        pool = ["a", "b", "c", "d", "e", "f", "g", "h"]
        first = bm._sample_with_replacement(pool, 10, seed=1)
        second = bm._sample_with_replacement(pool, 10, seed=2)
        assert first != second

    def test_returns_exactly_n_items_even_beyond_pool_size(self) -> None:
        pool = ["a", "b"]
        result = bm._sample_with_replacement(pool, 7, seed=42)
        assert len(result) == 7
        assert all(item in pool for item in result)

    def test_raises_on_empty_pool(self) -> None:
        with pytest.raises(ValueError, match="empty pool"):
            bm._sample_with_replacement([], 5, seed=42)


class TestBenchmarkValidation:
    """benchmark() input validation, mirroring measure_latency()'s."""

    def test_n_must_be_positive(self, both_backends_models_dir: Path) -> None:
        with pytest.raises(ValueError, match="n must be positive"):
            bm.benchmark(n=0, warmup=1, models_dir=both_backends_models_dir, data_path=FIXTURE_PATH)

    def test_warmup_must_be_non_negative(self, both_backends_models_dir: Path) -> None:
        with pytest.raises(ValueError, match="warmup must be >= 0"):
            bm.benchmark(
                n=5, warmup=-1, models_dir=both_backends_models_dir, data_path=FIXTURE_PATH
            )


class TestBenchmarkBothBackends:
    """benchmark(backends=('sklearn', 'onnx')) -- the full, official comparison."""

    def test_measures_both_backends_with_sane_percentiles(
        self, both_backends_models_dir: Path
    ) -> None:
        result = bm.benchmark(
            n=8, warmup=2, seed=42, models_dir=both_backends_models_dir, data_path=FIXTURE_PATH
        )

        assert set(result.backends) == {"sklearn", "onnx"}
        for name, r in result.backends.items():
            assert r.backend == name
            assert r.n == 8
            assert r.warmup == 2
            assert r.min_ms <= r.p50_ms <= r.p90_ms <= r.p95_ms <= r.p99_ms <= r.max_ms
            assert r.mean_ms >= 0.0
            assert r.throughput_rps > 0.0

    def test_computes_p50_reduction_when_both_backends_present(
        self, both_backends_models_dir: Path
    ) -> None:
        result = bm.benchmark(
            n=8, warmup=2, seed=42, models_dir=both_backends_models_dir, data_path=FIXTURE_PATH
        )

        assert result.p50_reduction_pct is not None
        assert result.p50_reduction_min_pct == bm.P50_REDUCTION_MIN_PCT
        assert result.p50_reduction_met == (result.p50_reduction_pct >= bm.P50_REDUCTION_MIN_PCT)

    def test_equivalence_uses_the_holdout_split_and_reports_real_numbers(
        self, both_backends_models_dir: Path
    ) -> None:
        result = bm.benchmark(
            n=5, warmup=1, seed=42, models_dir=both_backends_models_dir, data_path=FIXTURE_PATH
        )

        assert result.equivalence.n_compared > 0
        assert 0.0 <= result.equivalence.equivalence_pct <= 100.0
        assert result.equivalence.n_matches <= result.equivalence.n_compared
        assert result.equivalence.equivalence_min_pct == 99.0
        assert result.equivalence.passed == (
            result.equivalence.equivalence_pct >= result.equivalence.equivalence_min_pct
        )


class TestBenchmarkSingleBackend:
    """benchmark(backends=(...one backend...)) -- partial runs stay well-defined."""

    def test_sklearn_only_has_no_p50_reduction_and_no_equivalence(
        self, sklearn_only_models_dir: Path
    ) -> None:
        result = bm.benchmark(
            backends=("sklearn",),
            n=5,
            warmup=1,
            models_dir=sklearn_only_models_dir,
            data_path=FIXTURE_PATH,
        )

        assert set(result.backends) == {"sklearn"}
        assert result.p50_reduction_pct is None
        assert result.p50_reduction_met is False
        assert result.equivalence.n_compared == 0
        assert result.equivalence.passed is True


class TestMachineInfoAndPayload:
    """_machine_info() / _result_to_payload() -- required by the honesty rule."""

    def test_machine_info_has_required_fields(self) -> None:
        machine = bm._machine_info()
        for key in (
            "platform",
            "processor",
            "python_version",
            "cpu_count",
            "scikit_learn_version",
            "onnxruntime_version",
            "skl2onnx_version",
            "numpy_version",
        ):
            assert machine[key]

    def test_payload_round_trips_through_json(self, both_backends_models_dir: Path) -> None:
        result = bm.benchmark(
            n=5, warmup=1, models_dir=both_backends_models_dir, data_path=FIXTURE_PATH
        )
        payload = bm._result_to_payload(result, bm._machine_info())

        serialized = json.dumps(payload)  # must not raise
        reloaded = json.loads(serialized)
        assert reloaded["seed"] == result.seed
        assert set(reloaded["backends"]) == {"sklearn", "onnx"}


class TestRenderMarkdown:
    """_render_markdown() -- the human-readable docs/latency_report.md body."""

    def test_reports_p50_reduction_honestly_when_not_met(self) -> None:
        payload = {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "machine": bm._machine_info(),
            "seed": 42,
            "holdout_test_size": 0.2,
            "backends": {
                "sklearn": {
                    "backend": "sklearn",
                    "n": 200,
                    "warmup": 20,
                    "mean_ms": 1.0,
                    "p50_ms": 1.0,
                    "p90_ms": 1.5,
                    "p95_ms": 1.8,
                    "p99_ms": 2.0,
                    "throughput_rps": 900.0,
                },
                "onnx": {
                    "backend": "onnx",
                    "n": 200,
                    "warmup": 20,
                    "mean_ms": 0.95,
                    "p50_ms": 0.95,
                    "p90_ms": 1.4,
                    "p95_ms": 1.7,
                    "p99_ms": 1.9,
                    "throughput_rps": 950.0,
                },
            },
            "equivalence": {
                "n_compared": 100,
                "n_matches": 100,
                "equivalence_pct": 100.0,
                "equivalence_min_pct": 99.0,
                "passed": True,
            },
            "p50_reduction_pct": 5.0,
            "p50_reduction_min_pct": 20.0,
            "p50_reduction_met": False,
        }

        markdown = bm._render_markdown(payload)

        assert "5.00%" in markdown
        assert "NAO ATINGIDA" in markdown
        assert "reportado sem maquiagem" in markdown
        assert "100.0000%" in markdown or "100.00" in markdown


class TestCli:
    """python -m triagem.optimization.benchmark."""

    def test_main_writes_json_and_markdown_and_returns_zero_on_pass(
        self, tmp_path: Path, both_backends_models_dir: Path
    ) -> None:
        json_out = tmp_path / "comparison.json"
        md_out = tmp_path / "latency_report.md"

        exit_code = bm.main(
            [
                "--n",
                "5",
                "--warmup",
                "1",
                "--models-dir",
                str(both_backends_models_dir),
                "--data-path",
                str(FIXTURE_PATH),
                "--json-out",
                str(json_out),
                "--md-out",
                str(md_out),
            ]
        )

        assert json_out.exists()
        assert md_out.exists()
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        assert set(payload["backends"]) == {"sklearn", "onnx"}
        markdown = md_out.read_text(encoding="utf-8")
        assert "# Relatorio de latencia" in markdown
        # Whatever the real fixture-scale numbers are, exit code must reflect
        # the equivalence gate, not be hardcoded -- assert it matches.
        expected_exit = 0 if payload["equivalence"]["passed"] else 1
        assert exit_code == expected_exit

    def test_main_returns_one_when_equivalence_gate_fails(
        self, tmp_path: Path, both_backends_models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        json_out = tmp_path / "comparison.json"
        md_out = tmp_path / "latency_report.md"

        failing_result = bm.BenchmarkResult(
            backends={
                "sklearn": bm.BackendLatency(
                    backend="sklearn",
                    n=5,
                    warmup=1,
                    seed=42,
                    mean_ms=1.0,
                    stdev_ms=0.1,
                    min_ms=0.8,
                    p50_ms=1.0,
                    p90_ms=1.2,
                    p95_ms=1.3,
                    p99_ms=1.4,
                    max_ms=1.5,
                    throughput_rps=900.0,
                ),
                "onnx": bm.BackendLatency(
                    backend="onnx",
                    n=5,
                    warmup=1,
                    seed=42,
                    mean_ms=0.9,
                    stdev_ms=0.1,
                    min_ms=0.7,
                    p50_ms=0.9,
                    p90_ms=1.1,
                    p95_ms=1.2,
                    p99_ms=1.3,
                    max_ms=1.4,
                    throughput_rps=950.0,
                ),
            },
            equivalence=bm.EquivalenceResult(
                n_compared=10,
                n_matches=8,
                equivalence_pct=80.0,
                equivalence_min_pct=99.0,
                passed=False,
            ),
            p50_reduction_pct=10.0,
            p50_reduction_min_pct=20.0,
            p50_reduction_met=False,
            seed=42,
            measured_at="2026-01-01T00:00:00+00:00",
        )
        monkeypatch.setattr(bm, "benchmark", lambda **kwargs: failing_result)

        exit_code = bm.main(
            [
                "--models-dir",
                str(both_backends_models_dir),
                "--data-path",
                str(FIXTURE_PATH),
                "--json-out",
                str(json_out),
                "--md-out",
                str(md_out),
            ]
        )

        assert exit_code == 1

    def test_build_arg_parser_defaults(self) -> None:
        args = bm._build_arg_parser().parse_args([])

        assert args.n == 200
        assert args.warmup == 20
        assert args.seed == 42
        assert args.backends == "sklearn,onnx"
        assert args.models_dir is None
        assert args.data_path is None
