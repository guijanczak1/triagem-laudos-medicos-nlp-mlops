"""FastAPI serving layer for the triage classifier (T10).

Exposes ``POST /predict``, ``POST /predict/batch``, ``GET /health``,
``GET /model-info`` and ``GET /metrics`` (Prometheus instrumentation, T17
-- see :mod:`triagem.serving.metrics`) on top of the backend-agnostic
``Predictor`` (T9). The model is loaded exactly once, in the ASGI
lifespan (startup) -- never per request: ``Predictor.load()`` is called a
single time in :func:`_lifespan` and the resulting instance is stashed on
``app.state.predictor`` for every handler to reuse (on top of
``Predictor``'s own lazy-singleton cache).

If the configured backend's artifact is missing at startup, the process
still starts (so a container orchestrator can see it and `/health` can
report the problem) but ``app.state.predictor`` stays ``None``; every
endpoint that needs a model then responds ``503`` instead of crashing.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from triagem.config import get_settings
from triagem.exceptions import ModelArtifactNotFound
from triagem.serving import metrics as prom_metrics
from triagem.serving.metrics import PrometheusMiddleware
from triagem.serving.predictor import Predictor
from triagem.serving.schemas import (
    LABEL_PT,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictionItem,
    PredictRequest,
    PredictResponse,
)
from triagem.training.train import METRICS_FILENAME

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model exactly once at startup; never per request."""
    app.state.startup_time = time.monotonic()
    try:
        app.state.predictor = Predictor.load()
    except ModelArtifactNotFound as exc:
        logger.warning("model failed to load at startup, endpoints will 503: %s", exc)
        app.state.predictor = None
    else:
        meta = app.state.predictor.metadata
        prom_metrics.set_model_info(backend=meta.backend, model_version=meta.model_version)
    yield


def _get_predictor(request: Request) -> Predictor:
    """FastAPI dependency: the startup-loaded Predictor, or a 503 if absent."""
    predictor: Predictor | None = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded; check GET /health.")
    return predictor


def _read_metrics(models_dir: Path) -> dict[str, Any] | None:
    """Return the parsed contents of ``models/metrics.json``, or ``None`` if absent."""
    metrics_path = models_dir / METRICS_FILENAME
    if not metrics_path.exists():
        return None
    return dict(json.loads(metrics_path.read_text(encoding="utf-8")))


def create_app() -> FastAPI:
    """Build the FastAPI application.

    A factory (rather than a single module-level instance) keeps the app
    trivially re-constructible in tests that need a clean lifespan run
    against a fixture model directory.
    """
    app = FastAPI(
        title="Triagem API",
        description=(
            "Assistive clinical-report urgency triage. Predicts one of "
            "'normal', 'atencao' or 'urgente' for free-text medical reports. "
            "The urgency label is a didactic heuristic, not a validated "
            "clinical decision tool -- see docs/model_card.md."
        ),
        version="0.1.0",
        lifespan=_lifespan,
    )

    # Records triagem_requests_total / triagem_request_duration_seconds /
    # triagem_errors_total for every route below; never instruments GET
    # /metrics itself (T17).
    app.add_middleware(PrometheusMiddleware)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        body = ErrorResponse(detail=exc.errors(), error_type="validation_error")
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(
        _request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        body = ErrorResponse(detail=exc.detail, error_type="http_error")
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(_request: Request, _exc: Exception) -> JSONResponse:
        logger.exception("unhandled error while serving request")
        body = ErrorResponse(detail="Internal server error.", error_type="internal_error")
        return JSONResponse(status_code=500, content=body.model_dump())

    @app.get(
        "/metrics",
        tags=["ops"],
        summary="Prometheus metrics (text exposition format)",
        include_in_schema=False,
    )
    async def metrics_endpoint() -> Response:
        return Response(content=prom_metrics.render_latest(), media_type=prom_metrics.CONTENT_TYPE)

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse}},
        tags=["ops"],
        summary="Liveness/readiness probe",
    )
    async def health(request: Request) -> JSONResponse:
        predictor: Predictor | None = request.app.state.predictor
        uptime_s = time.monotonic() - request.app.state.startup_time
        if predictor is None:
            settings = get_settings()
            body = HealthResponse(
                status="error",
                model_backend=settings.model_backend,
                model_version="unknown",
                uptime_s=uptime_s,
            )
            return JSONResponse(status_code=503, content=body.model_dump())
        body = HealthResponse(
            status="ok",
            model_backend=predictor.metadata.backend,
            model_version=predictor.metadata.model_version,
            uptime_s=uptime_s,
        )
        return JSONResponse(status_code=200, content=body.model_dump())

    @app.get(
        "/model-info",
        response_model=ModelInfoResponse,
        responses={503: {"model": ErrorResponse}},
        tags=["ops"],
        summary="Loaded model metadata and training metrics",
    )
    async def model_info(predictor: Predictor = Depends(_get_predictor)) -> ModelInfoResponse:
        metrics = _read_metrics(get_settings().models_dir)
        return ModelInfoResponse(
            backend=predictor.metadata.backend,
            model_version=predictor.metadata.model_version,
            trained_at=predictor.metadata.trained_at,
            classes=list(predictor.metadata.classes),
            metrics=metrics,
        )

    @app.post(
        "/predict",
        response_model=PredictResponse,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        tags=["inference"],
        summary="Triage a single report",
    )
    async def predict(
        payload: PredictRequest, predictor: Predictor = Depends(_get_predictor)
    ) -> PredictResponse:
        result = predictor.predict(payload.text)
        prom_metrics.record_prediction(
            label=result.label, backend=result.backend, latency_ms=result.latency_ms
        )
        return PredictResponse(
            label=result.label,
            label_pt=LABEL_PT[result.label],
            scores=result.scores,
            latency_ms=result.latency_ms,
            backend=result.backend,
            model_version=predictor.metadata.model_version,
        )

    @app.post(
        "/predict/batch",
        response_model=PredictBatchResponse,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        tags=["inference"],
        summary="Triage up to 64 reports in one call",
    )
    async def predict_batch(
        payload: PredictBatchRequest, predictor: Predictor = Depends(_get_predictor)
    ) -> PredictBatchResponse:
        start = time.perf_counter()
        results = predictor.predict_batch(payload.texts)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        for result in results:
            prom_metrics.record_prediction(
                label=result.label, backend=result.backend, latency_ms=result.latency_ms
            )
        predictions = [
            PredictionItem(
                label=result.label,
                label_pt=LABEL_PT[result.label],
                scores=result.scores,
                latency_ms=result.latency_ms,
                backend=result.backend,
            )
            for result in results
        ]
        return PredictBatchResponse(
            predictions=predictions, latency_ms=elapsed_ms, backend=predictor.metadata.backend
        )

    return app


#: Module-level ASGI app -- the entrypoint uvicorn/Docker (T11) point at,
#: e.g. ``uvicorn triagem.serving.api:app``.
app = create_app()
