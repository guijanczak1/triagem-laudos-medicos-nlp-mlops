"""Tests for triagem.serving.api.

Mirrors tests/serving/test_predictor.py's approach: trains a real (small)
pipeline in memory on the deterministic ``tests/fixtures/laudos_sample.csv``
fixture and persists it as a genuine ``model.joblib`` under ``tmp_path``, so
the API's lifespan loads a real ``Predictor`` -- no artifact on the real
``models/`` directory is required or touched. Uses ``fastapi.testclient``
so lifespan (startup/shutdown) actually runs, per T10's "loaded once at
startup, not per request" requirement.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from triagem.config import get_settings
from triagem.data.loaders import VALID_LABELS, load_dataset
from triagem.models.pipeline import build_pipeline
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
    (models_dir / METRICS_FILENAME).write_text(json.dumps(METRICS_PAYLOAD), encoding="utf-8")
    (models_dir / LABEL_ENCODER_FILENAME).write_text(
        json.dumps({"classes": sorted(VALID_LABELS)}), encoding="utf-8"
    )
    return models_dir


@pytest.fixture
def client(sklearn_models_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A TestClient whose app lifespan loads a real sklearn Predictor at startup."""
    monkeypatch.setenv("TRIAGEM_MODELS_DIR", str(sklearn_models_dir))
    monkeypatch.setenv("TRIAGEM_MODEL_BACKEND", "sklearn")
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A TestClient whose app lifespan fails to find any model artifact."""
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    monkeypatch.setenv("TRIAGEM_MODELS_DIR", str(empty_dir))
    monkeypatch.setenv("TRIAGEM_MODEL_BACKEND", "sklearn")
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client


class TestHealth:
    """GET /health."""

    def test_health_returns_200_ok_when_model_loaded(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_backend"] == "sklearn"
        assert body["model_version"] == TRAINED_AT
        assert body["uptime_s"] >= 0.0

    def test_health_returns_503_when_model_failed_to_load(
        self, client_without_model: TestClient
    ) -> None:
        response = client_without_model.get("/health")

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "error"
        assert body["model_backend"] == "sklearn"
        assert body["model_version"] == "unknown"


class TestPredict:
    """POST /predict."""

    def test_predict_returns_200_with_expected_contract(self, client: TestClient) -> None:
        response = client.post("/predict", json={"text": SAMPLE_TEXT})

        assert response.status_code == 200
        body = response.json()
        assert body["label"] in VALID_LABELS
        assert body["label_pt"] == LABEL_PT[body["label"]]
        assert set(body["scores"]) == set(VALID_LABELS)
        assert body["scores"][body["label"]] == max(body["scores"].values())
        assert sum(body["scores"].values()) == pytest.approx(1.0, abs=1e-6)
        assert body["latency_ms"] >= 0.0
        assert body["backend"] == "sklearn"
        assert body["model_version"] == TRAINED_AT

    def test_predict_empty_text_returns_422_standardized_body(self, client: TestClient) -> None:
        response = client.post("/predict", json={"text": ""})

        assert response.status_code == 422
        body = response.json()
        assert body["error_type"] == "validation_error"
        assert "detail" in body

    def test_predict_missing_field_returns_422(self, client: TestClient) -> None:
        response = client.post("/predict", json={})

        assert response.status_code == 422
        assert response.json()["error_type"] == "validation_error"

    def test_predict_text_over_max_length_returns_422(self, client: TestClient) -> None:
        response = client.post("/predict", json={"text": "a" * 20001})

        assert response.status_code == 422
        assert response.json()["error_type"] == "validation_error"

    def test_predict_returns_503_when_model_not_loaded(
        self, client_without_model: TestClient
    ) -> None:
        response = client_without_model.post("/predict", json={"text": SAMPLE_TEXT})

        assert response.status_code == 503
        body = response.json()
        assert body["error_type"] == "http_error"
        assert "detail" in body

    def test_predict_internal_error_returns_500_without_traceback(
        self, sklearn_models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Starlette's ServerErrorMiddleware always re-raises the original
        # exception after invoking our handler (so servers can log it / test
        # clients can opt into raising it) -- raise_server_exceptions=False
        # is required here to observe the actual HTTP response our handler
        # sent, instead of pytest re-raising RuntimeError itself.
        monkeypatch.setenv("TRIAGEM_MODELS_DIR", str(sklearn_models_dir))
        monkeypatch.setenv("TRIAGEM_MODEL_BACKEND", "sklearn")
        get_settings.cache_clear()

        def _boom(_self: Predictor, _text: str) -> None:
            raise RuntimeError("simulated backend crash")

        monkeypatch.setattr(Predictor, "predict", _boom)

        with TestClient(app, raise_server_exceptions=False) as no_raise_client:
            response = no_raise_client.post("/predict", json={"text": SAMPLE_TEXT})

        assert response.status_code == 500
        assert response.json() == {
            "detail": "Internal server error.",
            "error_type": "internal_error",
        }
        assert "Traceback" not in response.text
        assert "simulated backend crash" not in response.text


class TestPredictBatch:
    """POST /predict/batch."""

    def test_predict_batch_returns_200_one_prediction_per_text(self, client: TestClient) -> None:
        texts = [
            SAMPLE_TEXT,
            "Chronic mild stable follow-up, routine screening.",
            "Malignant metastatic carcinoma with sepsis and shock.",
        ]

        response = client.post("/predict/batch", json={"texts": texts})

        assert response.status_code == 200
        body = response.json()
        assert body["backend"] == "sklearn"
        assert body["latency_ms"] >= 0.0
        assert len(body["predictions"]) == len(texts)
        for item in body["predictions"]:
            assert item["label"] in VALID_LABELS
            assert item["label_pt"] == LABEL_PT[item["label"]]
            assert sum(item["scores"].values()) == pytest.approx(1.0, abs=1e-6)

    def test_predict_batch_empty_list_returns_422(self, client: TestClient) -> None:
        response = client.post("/predict/batch", json={"texts": []})

        assert response.status_code == 422
        assert response.json()["error_type"] == "validation_error"

    def test_predict_batch_over_64_texts_returns_422(self, client: TestClient) -> None:
        response = client.post("/predict/batch", json={"texts": [SAMPLE_TEXT] * 65})

        assert response.status_code == 422
        assert response.json()["error_type"] == "validation_error"

    def test_predict_batch_returns_503_when_model_not_loaded(
        self, client_without_model: TestClient
    ) -> None:
        response = client_without_model.post("/predict/batch", json={"texts": [SAMPLE_TEXT]})

        assert response.status_code == 503
        assert response.json()["error_type"] == "http_error"


class TestModelInfo:
    """GET /model-info."""

    def test_model_info_returns_200_with_metadata_and_metrics(self, client: TestClient) -> None:
        response = client.get("/model-info")

        assert response.status_code == 200
        body = response.json()
        assert body["backend"] == "sklearn"
        assert body["model_version"] == TRAINED_AT
        assert body["trained_at"] == TRAINED_AT
        assert set(body["classes"]) == set(VALID_LABELS)
        assert body["metrics"]["macro_f1"] == pytest.approx(METRICS_PAYLOAD["macro_f1"])
        assert body["metrics"]["recall_urgente"] == pytest.approx(METRICS_PAYLOAD["recall_urgente"])

    def test_model_info_returns_503_when_model_not_loaded(
        self, client_without_model: TestClient
    ) -> None:
        response = client_without_model.get("/model-info")

        assert response.status_code == 503
        assert response.json()["error_type"] == "http_error"


class TestOpenApiDocs:
    """OpenAPI schema must be reachable and carry filled-in examples."""

    def test_docs_endpoint_is_reachable(self, client: TestClient) -> None:
        response = client.get("/docs")

        assert response.status_code == 200

    def test_openapi_schema_has_filled_examples_for_predict_bodies(
        self, client: TestClient
    ) -> None:
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()
        components = schema["components"]["schemas"]
        assert "example" in components["PredictRequest"]
        assert components["PredictRequest"]["example"]["text"]
        assert "example" in components["PredictBatchRequest"]
        assert components["PredictBatchRequest"]["example"]["texts"]
