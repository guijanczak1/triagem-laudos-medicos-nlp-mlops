"""Tests for the API's onnx-by-default / explicit-sklearn-fallback behavior (T22).

Complements ``tests/serving/test_api.py`` (which only ever exercises the
``sklearn`` backend): this module trains the real T6 pipeline on the small
``tests/fixtures/laudos_sample.csv`` fixture, exports it to a real ONNX
graph with the real T20 exporter, and then drives the same ``/predict``
battery through ``fastapi.testclient.TestClient`` against **both** backends
-- plus the explicit-fallback path ``triagem.serving.api._lifespan`` adds on
top of ``Predictor`` (which itself never falls back; see T9/T20's own
tests). No mocking of the model or of onnxruntime: every assertion here runs
against a genuine artifact.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from triagem.config import get_settings
from triagem.data.loaders import VALID_LABELS, load_dataset
from triagem.models.pipeline import CLASSIFIER_STEP, build_pipeline
from triagem.optimization.onnx_export import export_to_onnx
from triagem.serving.api import app
from triagem.serving.predictor import Predictor
from triagem.serving.schemas import LABEL_PT
from triagem.training.train import LABEL_ENCODER_FILENAME, METRICS_FILENAME, MODEL_FILENAME

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "laudos_sample.csv"
TRAINED_AT = "2024-01-01T00:00:00+00:00"
SAMPLE_TEXT = "Acute severe emergency with critical organ failure and hemorrhage."
METRICS_PAYLOAD = {
    "trained_at": TRAINED_AT,
    "macro_f1": 0.6427,
    "recall_urgente": 0.7364,
    "accuracy": 0.6343,
    "seed": 42,
    "n_samples": 14438,
}


@pytest.fixture(autouse=True)
def _clear_caches() -> Iterator[None]:
    """Every test gets a fresh Settings + Predictor cache, and leaves none behind."""
    Predictor.clear_cache()
    get_settings.cache_clear()
    yield
    Predictor.clear_cache()
    get_settings.cache_clear()


@pytest.fixture(scope="module")
def _fitted_pipeline() -> Pipeline:
    """Fit the real T6/T7 pipeline on the small fixture once per test module."""
    df = load_dataset(FIXTURE_PATH)
    pipeline = build_pipeline(seed=42)
    pipeline.fit(df["text"].tolist(), df["label"].tolist())
    return pipeline


def _write_sklearn_sidecars(models_dir: Path, pipeline: Pipeline) -> None:
    (models_dir / METRICS_FILENAME).write_text(json.dumps(METRICS_PAYLOAD), encoding="utf-8")
    classifier = pipeline.named_steps[CLASSIFIER_STEP]
    (models_dir / LABEL_ENCODER_FILENAME).write_text(
        json.dumps({"classes": [str(c) for c in classifier.classes_]}), encoding="utf-8"
    )


@pytest.fixture
def sklearn_models_dir(tmp_path: Path, _fitted_pipeline: Pipeline) -> Path:
    """A models/ dir with only a real sklearn artifact (no model.onnx)."""
    models_dir = tmp_path / "models_sklearn_only"
    models_dir.mkdir()
    joblib.dump(_fitted_pipeline, models_dir / MODEL_FILENAME)
    _write_sklearn_sidecars(models_dir, _fitted_pipeline)
    return models_dir


@pytest.fixture
def both_backends_models_dir(tmp_path: Path, _fitted_pipeline: Pipeline) -> Path:
    """A models/ dir with real sklearn AND real onnx artifacts (T20's exporter)."""
    models_dir = tmp_path / "models_both"
    models_dir.mkdir()
    joblib.dump(_fitted_pipeline, models_dir / MODEL_FILENAME)
    _write_sklearn_sidecars(models_dir, _fitted_pipeline)
    export_to_onnx(models_dir / MODEL_FILENAME, models_dir / "model.onnx")
    return models_dir


def _client_for(
    models_dir: Path, backend: str | None, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    """A TestClient whose app lifespan runs for real against ``models_dir``.

    ``backend=None`` leaves ``TRIAGEM_MODEL_BACKEND`` unset, so
    ``Settings``'s own default (``"onnx"``) governs what the lifespan tries
    first -- used by the tests that assert the default itself.
    """
    monkeypatch.setenv("TRIAGEM_MODELS_DIR", str(models_dir))
    if backend is None:
        monkeypatch.delenv("TRIAGEM_MODEL_BACKEND", raising=False)
    else:
        monkeypatch.setenv("TRIAGEM_MODEL_BACKEND", backend)
    get_settings.cache_clear()
    return TestClient(app)


class TestPredictBothBackendsParametrized:
    """The full /predict contract must hold for both backends alike."""

    @pytest.mark.parametrize("backend", ["sklearn", "onnx"])
    def test_predict_returns_200_with_expected_contract(
        self,
        backend: str,
        both_backends_models_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _client_for(both_backends_models_dir, backend, monkeypatch) as client:
            response = client.post("/predict", json={"text": SAMPLE_TEXT})

        assert response.status_code == 200
        body = response.json()
        assert body["label"] in VALID_LABELS
        assert body["label_pt"] == LABEL_PT[body["label"]]
        assert set(body["scores"]) == set(VALID_LABELS)
        assert body["scores"][body["label"]] == max(body["scores"].values())
        assert sum(body["scores"].values()) == pytest.approx(1.0, abs=1e-6)
        assert body["latency_ms"] >= 0.0
        assert body["backend"] == backend

    @pytest.mark.parametrize("backend", ["sklearn", "onnx"])
    def test_predict_empty_text_returns_422_regardless_of_backend(
        self,
        backend: str,
        both_backends_models_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _client_for(both_backends_models_dir, backend, monkeypatch) as client:
            response = client.post("/predict", json={"text": ""})

        assert response.status_code == 422
        assert response.json()["error_type"] == "validation_error"

    @pytest.mark.parametrize("backend", ["sklearn", "onnx"])
    def test_predict_batch_returns_200_one_prediction_per_text(
        self,
        backend: str,
        both_backends_models_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        texts = [
            SAMPLE_TEXT,
            "Chronic mild stable follow-up, routine screening.",
            "Malignant metastatic carcinoma with sepsis and shock.",
        ]
        with _client_for(both_backends_models_dir, backend, monkeypatch) as client:
            response = client.post("/predict/batch", json={"texts": texts})

        assert response.status_code == 200
        body = response.json()
        assert body["backend"] == backend
        assert len(body["predictions"]) == len(texts)
        for item in body["predictions"]:
            assert item["backend"] == backend
            assert sum(item["scores"].values()) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("backend", ["sklearn", "onnx"])
    def test_health_reflects_the_effective_backend(
        self,
        backend: str,
        both_backends_models_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _client_for(both_backends_models_dir, backend, monkeypatch) as client:
            response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_backend"] == backend

    @pytest.mark.parametrize("backend", ["sklearn", "onnx"])
    def test_model_info_reflects_the_effective_backend(
        self,
        backend: str,
        both_backends_models_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _client_for(both_backends_models_dir, backend, monkeypatch) as client:
            response = client.get("/model-info")

        assert response.status_code == 200
        body = response.json()
        assert body["backend"] == backend
        assert set(body["classes"]) == set(VALID_LABELS)


class TestOnnxIsTheDefaultBackend:
    """With TRIAGEM_MODEL_BACKEND unset, the API must serve onnx by default."""

    def test_health_reports_onnx_when_backend_env_is_unset(
        self, both_backends_models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with _client_for(both_backends_models_dir, None, monkeypatch) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["model_backend"] == "onnx"

    def test_model_info_reports_onnx_when_backend_env_is_unset(
        self, both_backends_models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with _client_for(both_backends_models_dir, None, monkeypatch) as client:
            response = client.get("/model-info")

        assert response.status_code == 200
        assert response.json()["backend"] == "onnx"

    def test_predict_backend_is_onnx_when_backend_env_is_unset(
        self, both_backends_models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with _client_for(both_backends_models_dir, None, monkeypatch) as client:
            response = client.post("/predict", json={"text": SAMPLE_TEXT})

        assert response.status_code == 200
        assert response.json()["backend"] == "onnx"


class TestExplicitFallbackToSklearnWhenOnnxArtifactMissing:
    """models/model.onnx absent + default 'onnx' backend -> explicit, logged
    fallback to 'sklearn' -- never a silent one, never a 503 while a working
    sklearn artifact sits right there."""

    def test_health_falls_back_to_sklearn_and_stays_ok(
        self,
        sklearn_models_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _client_for(sklearn_models_dir, None, monkeypatch) as client:
            response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_backend"] == "sklearn"

    def test_model_info_falls_back_to_sklearn(
        self,
        sklearn_models_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _client_for(sklearn_models_dir, None, monkeypatch) as client:
            response = client.get("/model-info")

        assert response.status_code == 200
        assert response.json()["backend"] == "sklearn"

    def test_predict_still_works_via_the_fallback_backend(
        self,
        sklearn_models_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _client_for(sklearn_models_dir, None, monkeypatch) as client:
            response = client.post("/predict", json={"text": SAMPLE_TEXT})

        assert response.status_code == 200
        assert response.json()["backend"] == "sklearn"

    def test_metrics_endpoint_model_info_gauge_reports_the_fallback_backend(
        self,
        sklearn_models_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _client_for(sklearn_models_dir, None, monkeypatch) as client:
            body = client.get("/metrics").text

        assert 'triagem_model_info{backend="sklearn"' in body

    def test_fallback_logs_an_explicit_warning_naming_both_backends(
        self,
        sklearn_models_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with (
            caplog.at_level(logging.WARNING, logger="triagem.serving.api"),
            _client_for(sklearn_models_dir, None, monkeypatch),
        ):
            pass

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("onnx" in msg and "sklearn" in msg for msg in warnings), warnings

    def test_explicit_onnx_request_does_not_silently_fall_back_at_predictor_level(
        self, sklearn_models_dir: Path
    ) -> None:
        """T22's fallback is an API-startup policy, not a Predictor change --
        Predictor.load(backend='onnx') on its own must still fail loudly,
        exactly as T9/T20 established."""
        from triagem.exceptions import ModelArtifactNotFound

        with pytest.raises(ModelArtifactNotFound):
            Predictor.load(backend="onnx", models_dir=sklearn_models_dir)


class TestNoArtifactAtAllStill503s:
    """Neither model.onnx nor model.joblib present -> 503, fallback exhausted."""

    def test_health_returns_503_when_no_backend_has_an_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        empty_dir = tmp_path / "empty_models"
        empty_dir.mkdir()

        with _client_for(empty_dir, None, monkeypatch) as client:
            response = client.get("/health")

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "error"
        assert body["model_backend"] == "onnx"
