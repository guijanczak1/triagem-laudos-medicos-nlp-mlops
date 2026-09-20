"""Backend-agnostic online predictor: same contract for ``sklearn`` and ``onnx``.

``Predictor`` is the single seam the API (T10) and any other online-serving
caller use to turn raw text into a triage prediction, regardless of which
inference backend actually runs the model. Callers depend only on
``Predictor.load()`` / ``predict()`` / ``predict_batch()`` / ``metadata`` --
never on how a given backend loads or scores.

Two backends are supported today:

- ``"sklearn"`` -- loads ``models/model.joblib`` (the fitted
  ``TfidfVectorizer`` + ``LogisticRegression`` pipeline from T6/T7) via
  ``joblib.load`` and calls its ``predict_proba``.
- ``"onnx"`` -- loads ``models/model.onnx`` via ``onnxruntime`` once task
  T20 has exported it. Until that artifact exists, selecting this backend
  fails loudly with :class:`~triagem.exceptions.ModelArtifactNotFound` and
  a clear instruction -- there is **no silent fallback** to ``sklearn``.

Each backend is an isolated strategy (:class:`_SklearnBackend`,
:class:`_OnnxBackend`) behind the private :class:`_BackendStrategy`
interface, so adding/changing one backend never touches the other or the
public :class:`Predictor` contract.

The model is loaded from disk at most once per ``(backend, models_dir)``
pair: :meth:`Predictor.load` caches the constructed :class:`Predictor` in a
process-wide, lazily-populated dict, so repeated calls (e.g. one per
request, if a caller does not hold onto the instance itself) never re-read
the artifact from disk.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from triagem.config import ModelBackend, get_settings
from triagem.exceptions import ModelArtifactNotFound
from triagem.models.pipeline import CLASSIFIER_STEP
from triagem.training.train import LABEL_ENCODER_FILENAME, METRICS_FILENAME, MODEL_FILENAME

logger = logging.getLogger(__name__)

#: Filenames of the ONNX backend's artifacts, produced by task T20
#: (``triagem.optimization.onnx_export``, not yet implemented at the time
#: this module was written -- see the ``ModelArtifactNotFound`` message
#: below).
ONNX_MODEL_FILENAME = "model.onnx"
ONNX_META_FILENAME = "model_onnx_meta.json"


@dataclass(frozen=True)
class Prediction:
    """One prediction, whichever backend produced it.

    Attributes:
        label: The predicted class (``argmax`` of ``scores``).
        scores: ``label -> probability`` for every class, summing to 1.0
            (tolerance ``1e-6``). Exactly the keys in
            ``triagem.data.loaders.VALID_LABELS``.
        latency_ms: Wall-clock inference time in milliseconds. For
            :meth:`Predictor.predict_batch`, this is the batched call's
            total elapsed time divided by the number of inputs (an
            amortized per-item cost, not an independently measured one --
            batching runs a single vectorized call under the hood).
        backend: The backend that produced this prediction (``"sklearn"``
            or ``"onnx"``).
    """

    label: str
    scores: dict[str, float]
    latency_ms: float
    backend: str


@dataclass(frozen=True)
class PredictorMetadata:
    """Static metadata about a loaded :class:`Predictor`'s model.

    Attributes:
        backend: The backend actually loaded (``"sklearn"`` or ``"onnx"``).
        model_version: A stable identifier for the loaded artifact. There
            is no MLflow model registry in this project (see
            ``docs/model_card.md``: gate intentionally skipped, not
            required by the assignment), so this is the artifact's
            ``trained_at`` timestamp (from ``metrics.json`` /
            ``model_onnx_meta.json``) when available, else ``"unknown"``.
        trained_at: Raw ``trained_at`` timestamp string from the artifact's
            metadata file, or ``None`` if unavailable.
        classes: The class labels the model predicts over, in the exact
            order ``scores`` dicts are built from.
    """

    backend: str
    model_version: str
    trained_at: str | None
    classes: tuple[str, ...]


class _BackendStrategy(ABC):
    """Isolated, backend-specific loading + scoring. Never used directly."""

    classes_: tuple[str, ...]
    model_version: str
    trained_at: str | None

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one row of class probabilities per text, columns = ``classes_``."""


class _SklearnBackend(_BackendStrategy):
    """Strategy for the ``sklearn`` backend: ``models/model.joblib`` (T7)."""

    def __init__(self, models_dir: Path) -> None:
        model_path = models_dir / MODEL_FILENAME
        if not model_path.exists():
            raise ModelArtifactNotFound(
                f"{model_path} not found. Run `python -m triagem.training.train` first "
                "(task T7) to produce models/model.joblib, or point Predictor.load(...) / "
                "TRIAGEM_MODELS_DIR at a directory that already has it."
            )

        self._pipeline = joblib.load(model_path)
        classifier = self._pipeline.named_steps[CLASSIFIER_STEP]
        self.classes_ = tuple(str(label) for label in classifier.classes_)

        metrics = _read_optional_json(models_dir / METRICS_FILENAME) or {}
        self.trained_at = str(metrics["trained_at"]) if "trained_at" in metrics else None
        self.model_version = self.trained_at or "unknown"

    def predict_proba(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        proba: Any = self._pipeline.predict_proba(list(texts))
        return [[float(value) for value in row] for row in proba]


class _OnnxBackend(_BackendStrategy):
    """Strategy for the ``onnx`` backend: ``models/model.onnx`` (task T20).

    Isolated from :class:`_SklearnBackend` on purpose: a missing ``.onnx``
    artifact must fail loudly and immediately, never fall back to
    ``sklearn`` silently. Task T20 (owner: latency) exports the artifact
    this backend loads; until it runs, selecting ``"onnx"`` always raises
    :class:`ModelArtifactNotFound` below.

    The scoring path assumes the standard ``skl2onnx`` convention for a
    classifier pipeline converted with ``zipmap`` enabled (the default):
    ``onnxruntime`` returns a probabilities output as a sequence of
    ``{class_label: probability}`` dicts, one per input row. T20 owns
    finalizing this contract against a real exported artifact.
    """

    def __init__(self, models_dir: Path, intra_op_num_threads: int | None = None) -> None:
        model_path = models_dir / ONNX_MODEL_FILENAME
        if not model_path.exists():
            raise ModelArtifactNotFound(
                f"{model_path} not found. The 'onnx' backend requires an exported model: run "
                "`python -m triagem.optimization.onnx_export` first (task T20) to generate "
                f"{ONNX_MODEL_FILENAME} from {MODEL_FILENAME}, or select the 'sklearn' backend "
                "instead (TRIAGEM_MODEL_BACKEND=sklearn, or Predictor.load(backend='sklearn')). "
                "There is no silent fallback to sklearn."
            )

        import onnxruntime as ort  # local import: only the onnx backend needs this dependency

        session_options = ort.SessionOptions()
        if intra_op_num_threads is not None:
            session_options.intra_op_num_threads = intra_op_num_threads
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._proba_output_name = self._session.get_outputs()[-1].name

        meta = _read_optional_json(models_dir / ONNX_META_FILENAME) or {}
        classes = meta.get("classes")
        self.classes_ = (
            tuple(str(label) for label in classes)
            if classes
            else _classes_from_label_encoder(models_dir)
        )
        self.trained_at = str(meta["trained_at"]) if "trained_at" in meta else None
        self.model_version = self.trained_at or "unknown"

    def predict_proba(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        import numpy as np

        inputs = np.array(list(texts), dtype=object).reshape(-1, 1)
        outputs = self._session.run([self._proba_output_name], {self._input_name: inputs})
        raw = outputs[0]

        # Standard skl2onnx zipmap output: a sequence of {class: probability} dicts.
        if raw and isinstance(raw[0], dict):
            return [[float(row[label]) for label in self.classes_] for row in raw]
        # Defensive fallback: a plain (n_rows, n_classes) array/sequence.
        return [[float(value) for value in row] for row in raw]


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    """Return the parsed JSON object at ``path``, or ``None`` if it does not exist."""
    if not path.exists():
        return None
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _classes_from_label_encoder(models_dir: Path) -> tuple[str, ...]:
    """Fall back to the sklearn artifact's ``label_encoder.json`` for class order."""
    payload = _read_optional_json(models_dir / LABEL_ENCODER_FILENAME)
    if not payload or "classes" not in payload:
        raise ModelArtifactNotFound(
            f"{models_dir / ONNX_META_FILENAME} is missing 'classes' and "
            f"{models_dir / LABEL_ENCODER_FILENAME} is unavailable as a fallback; cannot "
            "determine the class order for the 'onnx' backend."
        )
    return tuple(str(label) for label in payload["classes"])


def _build_strategy(backend: str, models_dir: Path) -> _BackendStrategy:
    """Construct the strategy for ``backend``. Raises for anything else."""
    if backend == "sklearn":
        return _SklearnBackend(models_dir)
    if backend == "onnx":
        return _OnnxBackend(models_dir)
    raise ValueError(f"Unknown model backend {backend!r}; expected 'sklearn' or 'onnx'.")


#: Process-wide cache of loaded predictors, keyed by ``(backend, models_dir)``.
#: Keeps ``Predictor.load()`` a true lazy singleton per configuration: the
#: artifact is read from disk at most once even if callers invoke ``load()``
#: repeatedly (e.g. once per request) instead of holding onto the instance.
_PREDICTOR_CACHE: dict[tuple[str, str], Predictor] = {}


class Predictor:
    """Backend-agnostic triage predictor. Construct via :meth:`load`, not directly."""

    def __init__(self, strategy: _BackendStrategy, backend: str) -> None:
        self._strategy = strategy
        self._backend = backend

    @classmethod
    def load(
        cls,
        backend: ModelBackend | None = None,
        models_dir: Path | None = None,
    ) -> Predictor:
        """Load (or reuse a cached) :class:`Predictor` for ``backend``.

        Args:
            backend: ``"sklearn"`` or ``"onnx"``. Defaults to
                ``Settings.model_backend`` when omitted.
            models_dir: Directory holding the model artifacts. Defaults to
                ``Settings.models_dir`` when omitted.

        Returns:
            A :class:`Predictor` ready to score text. The underlying model
            is read from disk at most once per distinct
            ``(backend, models_dir)`` pair for the lifetime of the process
            (see :data:`_PREDICTOR_CACHE`).

        Raises:
            ModelArtifactNotFound: The required artifact is missing for
                the requested backend (e.g. ``models/model.onnx`` before
                task T20 runs). Never falls back to another backend.
        """
        settings = get_settings()
        resolved_backend: ModelBackend = backend if backend is not None else settings.model_backend
        resolved_dir = models_dir if models_dir is not None else settings.models_dir

        cache_key = (resolved_backend, str(resolved_dir))
        cached = _PREDICTOR_CACHE.get(cache_key)
        if cached is not None:
            return cached

        strategy = _build_strategy(resolved_backend, resolved_dir)
        predictor = cls(strategy, resolved_backend)
        _PREDICTOR_CACHE[cache_key] = predictor
        logger.info("predictor loaded: backend=%s models_dir=%s", resolved_backend, resolved_dir)
        return predictor

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the process-wide predictor cache. Mainly useful for tests."""
        _PREDICTOR_CACHE.clear()

    @property
    def metadata(self) -> PredictorMetadata:
        """Static metadata about the loaded model (backend, version, classes, ...)."""
        return PredictorMetadata(
            backend=self._backend,
            model_version=self._strategy.model_version,
            trained_at=self._strategy.trained_at,
            classes=self._strategy.classes_,
        )

    def predict(self, text: str) -> Prediction:
        """Predict the triage label + class scores for a single text."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> list[Prediction]:
        """Predict for several texts in one vectorized backend call.

        Args:
            texts: Input texts. Empty input returns an empty list without
                touching the backend.

        Returns:
            One :class:`Prediction` per input text, same order as
            ``texts``.
        """
        text_list = list(texts)
        if not text_list:
            return []

        start = time.perf_counter()
        raw_proba = self._strategy.predict_proba(text_list)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_item_ms = elapsed_ms / len(text_list)

        classes = self._strategy.classes_
        predictions: list[Prediction] = []
        for row in raw_proba:
            scores = {label: float(p) for label, p in zip(classes, row, strict=True)}
            best_label = max(scores, key=lambda label: scores[label])
            predictions.append(
                Prediction(
                    label=best_label,
                    scores=scores,
                    latency_ms=per_item_ms,
                    backend=self._backend,
                )
            )
        return predictions
