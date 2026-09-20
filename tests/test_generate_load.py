"""Tests for scripts/generate_load.py (T19).

Same pattern as ``tests/test_measure_latency.py``: ``scripts/`` sits outside
the ``triagem`` package, so the module is loaded here by file path. No
network is used -- every ``requests`` call is monkeypatched, and the text
mix is drawn from the small, deterministic
``tests/fixtures/laudos_sample.csv`` fixture (3 classes, <= 60 rows) instead
of the full dataset.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from triagem.data.loaders import VALID_LABELS, load_dataset

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "laudos_sample.csv"
SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "generate_load.py"


def _load_generate_load_module() -> ModuleType:
    """Import scripts/generate_load.py by path (it lives outside the package)."""
    spec = importlib.util.spec_from_file_location("generate_load", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gl = _load_generate_load_module()


class TestBuildTextMix:
    """build_text_mix() -- deterministic, class-diverse sampling."""

    def test_returns_exactly_n_texts(self) -> None:
        texts = gl.build_text_mix(30, seed=42, data_path=FIXTURE_PATH)
        assert len(texts) == 30
        assert all(isinstance(t, str) and t for t in texts)

    def test_same_seed_yields_identical_mix(self) -> None:
        first = gl.build_text_mix(24, seed=42, data_path=FIXTURE_PATH)
        second = gl.build_text_mix(24, seed=42, data_path=FIXTURE_PATH)
        assert first == second

    def test_different_seed_yields_different_mix(self) -> None:
        first = gl.build_text_mix(24, seed=1, data_path=FIXTURE_PATH)
        second = gl.build_text_mix(24, seed=2, data_path=FIXTURE_PATH)
        assert first != second

    def test_draws_from_all_three_classes(self) -> None:
        # With n large relative to the fixture's ~45 rows and 3 balanced
        # classes, every class's texts must appear at least once in the mix.
        texts = set(gl.build_text_mix(60, seed=42, data_path=FIXTURE_PATH))
        df = load_dataset(FIXTURE_PATH)
        for label in sorted(VALID_LABELS):
            class_texts = set(df.loc[df["label"] == label, "text"].astype(str))
            assert texts & class_texts, f"no text from class {label!r} in the generated mix"

    def test_n_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n must be positive"):
            gl.build_text_mix(0, seed=42, data_path=FIXTURE_PATH)

    def test_n_smaller_than_class_count_still_returns_n(self) -> None:
        texts = gl.build_text_mix(2, seed=42, data_path=FIXTURE_PATH)
        assert len(texts) == 2


class TestWaitForHealth:
    """wait_for_health() -- polls GET /health, requests always mocked."""

    def test_true_immediately_when_health_returns_200(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import requests

        class _Resp:
            status_code = 200

        monkeypatch.setattr(requests, "get", lambda url, timeout: _Resp())
        assert gl.wait_for_health("http://127.0.0.1:8000", timeout_s=5.0, interval_s=0.01) is True

    def test_false_when_never_healthy_within_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import requests

        def _raise(url: str, timeout: float) -> None:
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(requests, "get", _raise)
        assert gl.wait_for_health("http://127.0.0.1:8000", timeout_s=0.05, interval_s=0.02) is False


class TestGenerateLoad:
    """generate_load() -- fires one POST /predict per text, never raises."""

    def test_all_ok_counts_every_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import requests

        class _Resp:
            def raise_for_status(self) -> None:
                return None

        posted: list[dict[str, Any]] = []

        def _fake_post(self: Any, url: str, json: dict[str, Any], timeout: float) -> _Resp:
            posted.append({"url": url, "json": json})
            return _Resp()

        monkeypatch.setattr(requests.Session, "post", _fake_post)
        texts = ["text one", "text two", "text three"]
        result = gl.generate_load("http://127.0.0.1:8000", texts)

        assert result.requested == 3
        assert result.ok == 3
        assert result.failed == 0
        assert len(posted) == 3
        assert posted[0]["url"] == "http://127.0.0.1:8000/predict"
        assert posted[0]["json"] == {"text": "text one"}

    def test_failures_are_counted_not_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import requests

        class _OkResp:
            def raise_for_status(self) -> None:
                return None

        calls = {"n": 0}

        def _flaky_post(self: Any, url: str, json: dict[str, Any], timeout: float) -> _OkResp:
            calls["n"] += 1
            if calls["n"] == 2:
                raise requests.ConnectionError("boom")
            return _OkResp()

        monkeypatch.setattr(requests.Session, "post", _flaky_post)
        result = gl.generate_load("http://127.0.0.1:8000", ["a", "b", "c"])

        assert result.requested == 3
        assert result.ok == 2
        assert result.failed == 1

    def test_throughput_is_non_negative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import requests

        class _Resp:
            def raise_for_status(self) -> None:
                return None

        monkeypatch.setattr(requests.Session, "post", lambda *a, **k: _Resp())
        result = gl.generate_load("http://127.0.0.1:8000", ["only one"])
        assert result.throughput_rps >= 0.0
        assert result.mean_latency_ms >= 0.0


class TestMain:
    """main() -- CLI wiring: wait, build mix, send, exit code."""

    def test_returns_1_when_health_never_comes_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import requests

        def _raise(url: str, timeout: float) -> None:
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(requests, "get", _raise)
        exit_code = gl.main(
            ["--n", "5", "--wait-timeout", "0.05", "--data-path", str(FIXTURE_PATH)]
        )
        assert exit_code == 1

    def test_skip_wait_sends_load_and_returns_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import requests

        class _Resp:
            def raise_for_status(self) -> None:
                return None

        monkeypatch.setattr(requests.Session, "post", lambda *a, **k: _Resp())
        exit_code = gl.main(
            [
                "--n",
                "6",
                "--skip-wait",
                "--data-path",
                str(FIXTURE_PATH),
            ]
        )
        assert exit_code == 0

    def test_all_requests_failing_returns_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import requests

        def _raise_post(self: Any, url: str, json: dict[str, Any], timeout: float) -> None:
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(requests.Session, "post", _raise_post)
        exit_code = gl.main(
            [
                "--n",
                "3",
                "--skip-wait",
                "--data-path",
                str(FIXTURE_PATH),
            ]
        )
        assert exit_code == 1
