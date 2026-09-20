"""Tests for triagem.serving.metrics (T17).

Uses the same "train a real small pipeline on the fixture CSV, persist it
under tmp_path, point Settings at it" approach as
``tests/serving/test_api.py``, so the API's lifespan loads a genuine
``Predictor`` and the ``PrometheusMiddleware`` / ``/metrics`` endpoint run
for real -- no mocking of prometheus_client itself.

``triagem.serving.metrics`` registers its Counters/Histograms/Gauge on a
module-level, process-wide ``CollectorRegistry`` (by design -- see the
module docstring), so values accumulate across tests/modules that share
this interpreter. Every assertion below therefore compares a **delta**
(value after an action minus value read just before it), never an
absolute value.
"""

from __future__ import annotations

import json
import re
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
from triagem.training.train import LABEL_ENCODER_FILENAME, METRICS_FILENAME, MODEL_FILENAME

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "laudos_sample.csv"
TRAINED_AT = "2024-01-01T00:00:00+00:00"
SAMPLE_TEXT = "Acute severe emergency with critical organ failure and hemorrhage."
METRICS_PAYLOAD = {"trained_at": TRAINED_AT, "macro_f1": 0.6427, "recall_urgente": 0.7364}

#: The 6 metric names T17 declares as a contract (T19 builds the Grafana
#: dashboard on these exact identifiers -- see backlog.json T17/T19).
EXPECTED_METRIC_NAMES = (
    "triagem_requests_total",
    "triagem_request_duration_seconds",
    "triagem_predictions_total",
    "triagem_errors_total",
    "triagem_inference_duration_seconds",
    "triagem_model_info",
)


def _metric_value(body: str, metric_name: str, **labels: str) -> float:
    """Sum the value(s) of every exposition line for ``metric_name`` whose
    labels are a superset of ``labels``. Returns 0.0 if none match.
    """
    pattern = re.compile(rf"^{re.escape(metric_name)}\{{([^}}]*)\}}\s+([0-9.eE+-]+)$", re.MULTILINE)
    total = 0.0
    found = False
    for label_str, value in pattern.findall(body):
        pairs = dict(item.split("=", 1) for item in label_str.split(",") if item)
        pairs = {k: v.strip('"') for k, v in pairs.items()}
        if all(pairs.get(k) == v for k, v in labels.items()):
            total += float(value)
            found = True
    return total if found else 0.0


def _lines_for(body: str, metric_name: str) -> list[str]:
    return [line for line in body.splitlines() if line.startswith(f"{metric_name}{{")]


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
    df = load_dataset(FIXTURE_PATH)
    pipeline = build_pipeline(seed=42)
    pipeline.fit(df["text"].tolist(), df["label"].tolist())
    return pipeline


@pytest.fixture
def sklearn_models_dir(tmp_path: Path, _fitted_pipeline: object) -> Path:
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
    monkeypatch.setenv("TRIAGEM_MODELS_DIR", str(sklearn_models_dir))
    monkeypatch.setenv("TRIAGEM_MODEL_BACKEND", "sklearn")
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client


class TestMetricsEndpoint:
    """GET /metrics."""

    def test_responds_200_text_plain_with_all_six_declared_metrics(
        self, client: TestClient
    ) -> None:
        response = client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
        for name in EXPECTED_METRIC_NAMES:
            assert name in response.text, f"{name} missing from /metrics body"

    def test_metrics_endpoint_never_instruments_itself(self, client: TestClient) -> None:
        client.get("/metrics")
        client.get("/metrics")
        body = client.get("/metrics").text

        for line in _lines_for(body, "triagem_requests_total"):
            assert 'endpoint="/metrics"' not in line
        for line in _lines_for(body, "triagem_request_duration_seconds_count"):
            assert 'endpoint="/metrics"' not in line

    def test_model_info_gauge_reflects_loaded_backend(self, client: TestClient) -> None:
        body = client.get("/metrics").text

        value = _metric_value(
            body,
            "triagem_model_info",
            backend="sklearn",
            model_version=TRAINED_AT,
            model_type="sklearn-pipeline",
        )
        assert value == 1.0


class TestPredictHitIncrementsCounters:
    """Bate em /predict e confere que os contadores em /metrics incrementaram."""

    def test_predict_increments_requests_predictions_and_inference_duration(
        self, client: TestClient
    ) -> None:
        before = client.get("/metrics").text

        response = client.post("/predict", json={"text": SAMPLE_TEXT})
        assert response.status_code == 200
        label = response.json()["label"]

        after = client.get("/metrics").text

        requests_before = _metric_value(
            before, "triagem_requests_total", endpoint="/predict", method="POST", status="200"
        )
        requests_after = _metric_value(
            after, "triagem_requests_total", endpoint="/predict", method="POST", status="200"
        )
        assert requests_after == requests_before + 1

        duration_count_before = _metric_value(
            before, "triagem_request_duration_seconds_count", endpoint="/predict"
        )
        duration_count_after = _metric_value(
            after, "triagem_request_duration_seconds_count", endpoint="/predict"
        )
        assert duration_count_after == duration_count_before + 1

        predictions_before = _metric_value(before, "triagem_predictions_total", label=label)
        predictions_after = _metric_value(after, "triagem_predictions_total", label=label)
        assert predictions_after == predictions_before + 1

        inference_count_before = _metric_value(
            before, "triagem_inference_duration_seconds_count", backend="sklearn"
        )
        inference_count_after = _metric_value(
            after, "triagem_inference_duration_seconds_count", backend="sklearn"
        )
        assert inference_count_after == inference_count_before + 1

    def test_predict_batch_increments_predictions_total_once_per_item(
        self, client: TestClient
    ) -> None:
        texts = [SAMPLE_TEXT, "Chronic mild stable follow-up, routine screening."]
        before = client.get("/metrics").text

        response = client.post("/predict/batch", json={"texts": texts})
        assert response.status_code == 200
        labels = [item["label"] for item in response.json()["predictions"]]

        after = client.get("/metrics").text

        for label in set(labels):
            expected_increment = labels.count(label)
            value_before = _metric_value(before, "triagem_predictions_total", label=label)
            value_after = _metric_value(after, "triagem_predictions_total", label=label)
            assert value_after == value_before + expected_increment

        requests_before = _metric_value(
            before,
            "triagem_requests_total",
            endpoint="/predict/batch",
            method="POST",
            status="200",
        )
        requests_after = _metric_value(
            after, "triagem_requests_total", endpoint="/predict/batch", method="POST", status="200"
        )
        assert requests_after == requests_before + 1


class TestErrorsTotal:
    """triagem_errors_total{endpoint,type} on 4xx/5xx responses."""

    def test_validation_error_increments_errors_total_as_client_error(
        self, client: TestClient
    ) -> None:
        before = client.get("/metrics").text

        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

        after = client.get("/metrics").text

        errors_before = _metric_value(
            before, "triagem_errors_total", endpoint="/predict", type="client_error"
        )
        errors_after = _metric_value(
            after, "triagem_errors_total", endpoint="/predict", type="client_error"
        )
        assert errors_after == errors_before + 1

        requests_before = _metric_value(
            before, "triagem_requests_total", endpoint="/predict", method="POST", status="422"
        )
        requests_after = _metric_value(
            after, "triagem_requests_total", endpoint="/predict", method="POST", status="422"
        )
        assert requests_after == requests_before + 1

    def test_unmatched_route_uses_fixed_endpoint_label_not_raw_path(
        self, client: TestClient
    ) -> None:
        """404s on arbitrary/probed paths must never leak into the endpoint label."""
        before = client.get("/metrics").text

        response = client.get("/this-path-does-not-exist-12345")
        assert response.status_code == 404

        after = client.get("/metrics").text

        unmatched_before = _metric_value(
            before, "triagem_requests_total", endpoint="unmatched", method="GET", status="404"
        )
        unmatched_after = _metric_value(
            after, "triagem_requests_total", endpoint="unmatched", method="GET", status="404"
        )
        assert unmatched_after == unmatched_before + 1
        assert "/this-path-does-not-exist-12345" not in after
