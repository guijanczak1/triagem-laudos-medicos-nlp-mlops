"""Online inference layer: the backend-agnostic Predictor (T9) and the API (T10)."""

from __future__ import annotations

from triagem.serving.api import app, create_app
from triagem.serving.predictor import Prediction, Predictor, PredictorMetadata

__all__ = ["Predictor", "Prediction", "PredictorMetadata", "app", "create_app"]
