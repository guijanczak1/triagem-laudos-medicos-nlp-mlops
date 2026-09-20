"""Prometheus instrumentation for the triage API (T17).

Exposes ``GET /metrics`` (wired in :mod:`triagem.serving.api`) with exactly
six metrics -- names are a **contract**: a future task (T19) builds the
Grafana dashboard directly on top of these exact identifiers, so none of
them may be renamed without updating the dashboard in the same commit
(see ``harness/observability.md``):

- ``triagem_requests_total{endpoint,method,status}`` (Counter) -- one HTTP
  request handled by the API.
- ``triagem_request_duration_seconds{endpoint}`` (Histogram) -- wall-clock
  time to serve one HTTP request, from the ASGI middleware.
- ``triagem_predictions_total{label}`` (Counter) -- one prediction made,
  by the predicted label.
- ``triagem_errors_total{endpoint,type}`` (Counter) -- one request that
  ended in a 4xx/5xx response.
- ``triagem_inference_duration_seconds{backend}`` (Histogram) -- model
  inference time (``Predictor.predict``/``predict_batch``), excluding
  HTTP/validation overhead, by backend.
- ``triagem_model_info{backend,model_version,model_type}`` (Gauge, always
  ``1``) -- static info about the currently loaded model, in the
  ``*_info`` convention (https://www.robustperception.io/exposing-the-software-version-to-prometheus).

Cardinality is controlled on purpose: ``endpoint`` is always the route's
**path template** (e.g. ``/predict``), never the raw request path, and no
label ever carries free text, an id or a timestamp.

Single-process assumption: this module registers metrics on a private
:class:`~prometheus_client.CollectorRegistry`, which is correct for a
single ``uvicorn`` worker (the default -- see T11's Dockerfile, no
``--workers`` flag). If a future task moves the API to multiple worker
processes (gunicorn/uvicorn ``--workers > 1``), counters would then be
inconsistent across workers unless the deployment switches to
``prometheus_client.multiprocess`` with ``PROMETHEUS_MULTIPROC_DIR`` set
before import -- not needed today, called out here per
``harness/observability.md``'s "regras duras".
"""

from __future__ import annotations

import time

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client import generate_latest as _generate_latest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: Path the middleware never instruments (avoids the endpoint measuring
#: its own scrape and inflating its own counters).
METRICS_PATH = "/metrics"

#: Bucket boundaries (seconds) for triagem_request_duration_seconds, per T17's
#: acceptance criteria -- tuned for a sub-second HTTP triage endpoint.
REQUEST_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

#: Fallback ``endpoint`` label for requests that never matched a route
#: (e.g. 404s from path scanning/probing) -- fixed value so arbitrary
#: request paths never become label values (cardinality control).
UNMATCHED_ENDPOINT = "unmatched"

#: Private registry (not the global default) so importing this module
#: repeatedly in tests/processes never raises prometheus_client's
#: "Duplicated timeseries" error and every metric exposed by /metrics is
#: declared explicitly here, nothing picked up implicitly.
REGISTRY = CollectorRegistry()

REQUESTS_TOTAL = Counter(
    "triagem_requests_total",
    "Total HTTP requests handled by the API.",
    ["endpoint", "method", "status"],
    registry=REGISTRY,
)

REQUEST_DURATION_SECONDS = Histogram(
    "triagem_request_duration_seconds",
    "HTTP request duration in seconds, from the ASGI middleware.",
    ["endpoint"],
    buckets=REQUEST_DURATION_BUCKETS,
    registry=REGISTRY,
)

PREDICTIONS_TOTAL = Counter(
    "triagem_predictions_total",
    "Total predictions made, by predicted label.",
    ["label"],
    registry=REGISTRY,
)

ERRORS_TOTAL = Counter(
    "triagem_errors_total",
    "Total requests that ended in a 4xx/5xx response, by endpoint and error type.",
    ["endpoint", "type"],
    registry=REGISTRY,
)

INFERENCE_DURATION_SECONDS = Histogram(
    "triagem_inference_duration_seconds",
    "Model inference duration in seconds (Predictor call only), by backend.",
    ["backend"],
    registry=REGISTRY,
)

MODEL_INFO = Gauge(
    "triagem_model_info",
    "Static info about the currently loaded model. Value is always 1.",
    ["backend", "model_version", "model_type"],
    registry=REGISTRY,
)

#: Low-cardinality, backend -> human-readable model type. Deliberately not
#: sourced from the sklearn/onnx internals (Predictor exposes no such
#: field, and T17 does not touch T9's predictor contract) -- a fixed map
#: is enough for an info-style gauge and stays cheap to extend.
_MODEL_TYPE_BY_BACKEND: dict[str, str] = {
    "sklearn": "sklearn-pipeline",
    "onnx": "onnx-runtime",
}


def render_latest() -> bytes:
    """Serialize all registered metrics in Prometheus text exposition format."""
    return _generate_latest(REGISTRY)


#: Exact ``Content-Type`` GET /metrics must respond with (text/plain;
#: version=0.0.4; charset=utf-8) -- re-exported so api.py never hardcodes it.
CONTENT_TYPE = CONTENT_TYPE_LATEST


def record_prediction(label: str, backend: str, latency_ms: float) -> None:
    """Record one prediction: increments ``triagem_predictions_total`` and
    observes ``triagem_inference_duration_seconds`` for ``backend``.

    ``latency_ms`` is the per-item inference latency already computed by
    :class:`triagem.serving.predictor.Predictor` (its own ``perf_counter``
    measurement around the backend call) -- reused here instead of timing
    again, so this stays a single source of truth for "how long did
    inference take".
    """
    PREDICTIONS_TOTAL.labels(label=label).inc()
    INFERENCE_DURATION_SECONDS.labels(backend=backend).observe(latency_ms / 1000.0)


def set_model_info(backend: str, model_version: str) -> None:
    """Set ``triagem_model_info`` for the model currently loaded at startup.

    Clears any previously-set label combination first: this gauge follows
    the ``*_info`` convention (always ``1``, all information in labels),
    so only one combination should ever be active per process -- without
    the clear, re-loading a different model in the same process (e.g.
    across tests that re-run the app lifespan) would leave stale rows
    behind in the exposition output.
    """
    MODEL_INFO.clear()
    model_type = _MODEL_TYPE_BY_BACKEND.get(backend, backend)
    MODEL_INFO.labels(backend=backend, model_version=model_version, model_type=model_type).set(1)


def _endpoint_label(scope: Scope) -> str:
    """Route **path template** for ``scope`` (e.g. ``/predict``), never the raw path.

    FastAPI's ``APIRoute.matches`` stashes the matched route on
    ``scope["route"]`` once routing succeeds (see
    ``fastapi.routing.APIRoute.matches``); ``route.path`` is the template
    Starlette compiled the route from, with path params unresolved (e.g.
    ``/items/{item_id}``, not ``/items/42``) -- exactly the low-cardinality
    label this module requires. Requests that never matched any route
    (404s from probing/scanning) have no such key; those collapse to the
    fixed ``UNMATCHED_ENDPOINT`` label instead of leaking arbitrary paths.
    """
    route = scope.get("route")
    path = getattr(route, "path", None)
    return str(path) if path else UNMATCHED_ENDPOINT


class PrometheusMiddleware:
    """Pure ASGI middleware recording per-request metrics for every route.

    Records ``triagem_requests_total`` and ``triagem_request_duration_seconds``
    for every HTTP request except ``GET /metrics`` itself (checked by raw
    ``scope["path"]``, before routing/templating happens, so the scrape
    endpoint never instruments itself). Also increments
    ``triagem_errors_total`` whenever the final response status is >= 400,
    or when the downstream app raises instead of producing a response at
    all (status recorded as ``"500"`` in that case, then the exception is
    re-raised unchanged for FastAPI's own exception handlers / ASGI server
    to deal with).

    Implemented as a plain ASGI callable (not ``BaseHTTPMiddleware``) per
    ``harness/observability.md``'s "Middleware ASGI" requirement -- it also
    avoids ``BaseHTTPMiddleware``'s extra request/response buffering, which
    is unnecessary overhead for something that only needs the status code
    and elapsed time.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] == METRICS_PATH:
            await self.app(scope, receive, send)
            return

        method = str(scope["method"])
        start = time.perf_counter()
        status_holder: list[int] = []

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder.append(int(message["status"]))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            endpoint = _endpoint_label(scope)
            elapsed = time.perf_counter() - start
            REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status="500").inc()
            REQUEST_DURATION_SECONDS.labels(endpoint=endpoint).observe(elapsed)
            ERRORS_TOTAL.labels(endpoint=endpoint, type="unhandled_exception").inc()
            raise

        elapsed = time.perf_counter() - start
        endpoint = _endpoint_label(scope)
        status = status_holder[0] if status_holder else 500
        REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status=str(status)).inc()
        REQUEST_DURATION_SECONDS.labels(endpoint=endpoint).observe(elapsed)
        if status >= 400:
            error_type = "client_error" if status < 500 else "server_error"
            ERRORS_TOTAL.labels(endpoint=endpoint, type=error_type).inc()
