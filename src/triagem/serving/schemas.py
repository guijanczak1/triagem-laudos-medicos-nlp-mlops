"""Pydantic request/response contracts for the triage serving API (T10).

Kept separate from :mod:`triagem.serving.api` so the wire contract can be
read, tested and imported (e.g. by client code or future frontends) without
pulling in FastAPI's routing machinery.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from triagem.data.loaders import VALID_LABELS

#: Human-readable Portuguese label for each machine label in ``VALID_LABELS``.
#: The API is documented in pt-BR (see HARNESS.md language convention); the
#: raw ``label`` stays the stable machine-readable value used elsewhere in
#: the codebase (training, metrics, thresholds).
LABEL_PT: dict[str, str] = {
    "normal": "Normal",
    "atencao": "Atenção",
    "urgente": "Urgente",
}

_EXAMPLE_TEXT = (
    "Patient presents with acute severe chest pain, hemorrhage and signs of "
    "critical organ failure; emergency evaluation required."
)


class PredictRequest(BaseModel):
    """Body of ``POST /predict``."""

    model_config = ConfigDict(json_schema_extra={"example": {"text": _EXAMPLE_TEXT}})

    text: str = Field(
        min_length=1,
        max_length=20000,
        description="Medical report / abstract text to triage.",
        examples=[_EXAMPLE_TEXT],
    )


class PredictBatchRequest(BaseModel):
    """Body of ``POST /predict/batch``."""

    model_config = ConfigDict(json_schema_extra={"example": {"texts": [_EXAMPLE_TEXT]}})

    texts: list[Annotated[str, Field(min_length=1, max_length=20000)]] = Field(
        min_length=1,
        max_length=64,
        description="1 to 64 medical report / abstract texts to triage.",
        examples=[[_EXAMPLE_TEXT]],
    )


class PredictionItem(BaseModel):
    """One prediction, whether standalone (``/predict``) or inside a batch."""

    label: str = Field(description="Predicted urgency class.", examples=[sorted(VALID_LABELS)[0]])
    label_pt: str = Field(description="Portuguese label for `label`.")
    scores: dict[str, float] = Field(
        description="`label -> probability` for every class, summing to 1.0."
    )
    latency_ms: float = Field(description="Inference wall-clock time in milliseconds.")
    backend: str = Field(description="Inference backend that produced this prediction.")


class PredictResponse(PredictionItem):
    """Response of ``POST /predict``."""

    model_version: str = Field(description="Stable identifier of the loaded model artifact.")


class PredictBatchResponse(BaseModel):
    """Response of ``POST /predict/batch``."""

    predictions: list[PredictionItem] = Field(
        description="One prediction per input text, same order as the request."
    )
    latency_ms: float = Field(description="Total wall-clock time for the whole batch call.")
    backend: str = Field(description="Inference backend that produced these predictions.")


class HealthResponse(BaseModel):
    """Response of ``GET /health``."""

    status: str = Field(description="'ok' if the model is loaded, 'error' otherwise.")
    model_backend: str = Field(description="Configured/loaded inference backend.")
    model_version: str = Field(description="Loaded model artifact version, or 'unknown'.")
    uptime_s: float = Field(description="Seconds since the API process started serving.")


class ModelInfoResponse(BaseModel):
    """Response of ``GET /model-info``."""

    backend: str = Field(description="Inference backend actually loaded.")
    model_version: str = Field(description="Stable identifier of the loaded model artifact.")
    trained_at: str | None = Field(description="Artifact training timestamp, if available.")
    classes: list[str] = Field(description="Class labels the model predicts over.")
    metrics: dict[str, Any] | None = Field(
        description="Raw contents of models/metrics.json, or null if unavailable."
    )


class ErrorResponse(BaseModel):
    """Standardized error body for 4xx/5xx responses."""

    detail: Any = Field(description="Human-readable error detail (string or structured list).")
    error_type: str = Field(description="Machine-readable error category.")
