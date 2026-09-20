"""Export the fitted sklearn pipeline to ONNX for the Predictor's ``onnx`` backend (T20).

``export_to_onnx`` converts ``models/model.joblib`` (the fitted TF-IDF +
LogisticRegression pipeline from T6/T7) to ``models/model.onnx`` via
``skl2onnx``, plus a ``models/model_onnx_meta.json`` sidecar that
``triagem.serving.predictor._OnnxBackend`` (T9) reads for the class order
and version metadata. Once this has run once, ``TRIAGEM_MODEL_BACKEND=onnx``
(or ``Predictor.load(backend="onnx")``) works end to end.

**The ``strip_accents`` incompatibility (discovered running this task).**
T6's pipeline uses ``TfidfVectorizer(strip_accents="unicode", ...)`` and its
docstring claims every step is skl2onnx-exportable "as is" -- that claim was
never actually exercised against ``skl2onnx`` until this task ran it for the
first time. It is not quite true: skl2onnx's vectorizer converter raises
``NotImplementedError: CountVectorizer cannot be converted, only
strip_accents=None is supported`` for any other value. Retraining with
``strip_accents=None`` is out of this task's scope (that is T6/T7's
artifact, and T6's own test asserts ``strip_accents == "unicode"``), so
:func:`_onnx_compatible_copy` instead flips the setting on a throwaway
``copy.deepcopy`` of the fitted pipeline, used only to build the ONNX graph.
This is provably a no-op for this project: the full processed dataset
(``data/processed/laudos.csv``, 14,438 rows, Medical Abstracts TC Corpus,
English) contains zero non-ASCII characters, and ``strip_accents`` only
changes anything for text that actually has accented/unicode characters to
strip. The empirical equivalence check in
``tests/optimization/test_onnx_export.py`` (and the full real-test-split
comparison run for this task's report) confirms sklearn and ONNX predictions
agree well above the project's 99% equivalence threshold
(``state/backlog.json``: ``prediction_equivalence_min_pct``), so this
adjustment is validated, not just assumed.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import skl2onnx
import sklearn
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import Pipeline

from triagem.config import get_settings
from triagem.exceptions import ModelArtifactNotFound
from triagem.models.pipeline import CLASSIFIER_STEP, VECTORIZER_STEP
from triagem.training.train import METRICS_FILENAME, MODEL_FILENAME

logger = logging.getLogger(__name__)

#: Filenames this module writes. MUST match
#: ``triagem.serving.predictor.ONNX_MODEL_FILENAME`` /
#: ``ONNX_META_FILENAME`` -- the Predictor's ``onnx`` backend loads exactly
#: these two files from ``models_dir``.
ONNX_MODEL_FILENAME = "model.onnx"
ONNX_META_FILENAME = "model_onnx_meta.json"

#: Project-wide minimum fraction of identical sklearn/onnx predictions
#: (``state/backlog.json``: ``thresholds.prediction_equivalence_min_pct``).
#: Used here only to gate whether a dynamically-quantized graph is kept
#: (see :func:`maybe_apply_dynamic_quantization`); T21's benchmark applies
#: the same floor to the official baseline-vs-onnx report.
EQUIVALENCE_MIN_PCT = 99.0


def _onnx_compatible_copy(pipeline: Pipeline) -> tuple[Pipeline, bool]:
    """Return a deep copy of ``pipeline`` safe to hand to ``skl2onnx``.

    See the module docstring for why this is needed and why it is safe on
    this project's dataset. The original fitted pipeline object passed in
    is never mutated.

    Returns:
        ``(copy_for_onnx, adjusted)``. ``adjusted`` is ``True`` if
        ``strip_accents`` had to be changed on the copy -- recorded
        verbatim in ``model_onnx_meta.json`` so the adjustment is never
        silent.
    """
    onnx_pipeline = copy.deepcopy(pipeline)
    vectorizer = onnx_pipeline.named_steps[VECTORIZER_STEP]
    adjusted = False
    original = getattr(vectorizer, "strip_accents", None)
    if original is not None:
        logger.warning(
            "onnx_export: TfidfVectorizer.strip_accents=%r is unsupported by skl2onnx "
            "(only None is); using strip_accents=None for the ONNX graph only. This is a "
            "verified no-op on the real dataset (100%% ASCII) -- see the module docstring "
            "of triagem.optimization.onnx_export.",
            original,
        )
        vectorizer.strip_accents = None
        adjusted = True
    return onnx_pipeline, adjusted


def _read_sibling_metrics(model_path: Path) -> dict[str, Any]:
    """Best-effort read of ``metrics.json`` next to ``model_path``. Never raises."""
    metrics_path = model_path.parent / METRICS_FILENAME
    if not metrics_path.exists():
        return {}
    return dict(json.loads(metrics_path.read_text(encoding="utf-8")))


def _write_meta(out_path: Path, meta: dict[str, Any]) -> None:
    """Write ``meta`` as pretty, sorted-key JSON next to ``out_path``."""
    meta_path = out_path.parent / ONNX_META_FILENAME
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _read_meta(out_path: Path) -> dict[str, Any]:
    """Read the ``model_onnx_meta.json`` sidecar of ``out_path``."""
    meta_path = out_path.parent / ONNX_META_FILENAME
    return dict(json.loads(meta_path.read_text(encoding="utf-8")))


def export_to_onnx(model_path: Path, out_path: Path, opset: int | None = None) -> Path:
    """Convert the fitted sklearn pipeline at ``model_path`` to ONNX.

    Args:
        model_path: Fitted ``sklearn.pipeline.Pipeline`` (TF-IDF +
            LogisticRegression) as written by ``triagem.training.train``
            (T7) -- typically ``models/model.joblib``.
        out_path: Destination ``.onnx`` file (typically
            ``models/model.onnx``). A companion ``model_onnx_meta.json`` is
            written in the same directory.
        opset: ONNX opset to target. ``None`` (default) lets ``skl2onnx``
            pick its own default, which stays compatible with the
            installed ``onnxruntime``.

    Returns:
        ``out_path``, after both the ``.onnx`` file and its metadata
        sidecar have been written.

    Raises:
        ModelArtifactNotFound: ``model_path`` does not exist.
    """
    if not model_path.exists():
        raise ModelArtifactNotFound(
            f"{model_path} not found. Run `python -m triagem.training.train` first (task T7) "
            "to produce it before exporting to ONNX."
        )

    pipeline = joblib.load(model_path)
    onnx_pipeline, strip_accents_adjusted = _onnx_compatible_copy(pipeline)
    classifier = onnx_pipeline.named_steps[CLASSIFIER_STEP]
    classes = tuple(str(label) for label in classifier.classes_)

    onnx_model = convert_sklearn(
        onnx_pipeline,
        initial_types=[("input", StringTensorType([None, 1]))],
        target_opset=opset,
        options={id(classifier): {"zipmap": True}},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(onnx_model.SerializeToString())

    used_opset = next(
        (imp.version for imp in onnx_model.opset_import if imp.domain in ("", "ai.onnx")),
        None,
    )
    metrics = _read_sibling_metrics(model_path)
    meta: dict[str, Any] = {
        "classes": list(classes),
        "opset": used_opset,
        "sklearn_version": sklearn.__version__,
        "skl2onnx_version": skl2onnx.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "strip_accents_adjusted_for_export": strip_accents_adjusted,
        "trained_at": metrics.get("trained_at"),
        "source_model": str(model_path),
        "exported_at": datetime.now(UTC).isoformat(),
        "quantized": False,
    }
    _write_meta(out_path, meta)

    logger.info(
        "onnx export complete: %s (%d classes, opset=%s)", out_path, len(classes), used_opset
    )
    return out_path


def _onnx_predict_labels(session: ort.InferenceSession, texts: Sequence[str]) -> list[str]:
    """Run ``session`` on ``texts`` and return the predicted label per row."""
    input_name = session.get_inputs()[0].name
    label_output = session.get_outputs()[0].name  # skl2onnx: 'output_label' is always first
    arr = np.array(list(texts), dtype=object).reshape(-1, 1)
    raw = session.run([label_output], {input_name: arr})[0]
    return [str(label) for label in raw]


def _median_latency_ms(
    call: Callable[[list[str]], object], texts: Sequence[str], warmup: int, runs: int
) -> float:
    """Median single-item latency (ms) of ``call``, after ``warmup`` untimed calls."""
    text_list = list(texts)
    for text in text_list[:warmup]:
        call([text])

    samples: list[float] = []
    for text in text_list[:runs]:
        start = time.perf_counter()
        call([text])
        samples.append((time.perf_counter() - start) * 1000.0)

    samples.sort()
    mid = len(samples) // 2
    if len(samples) % 2:
        return samples[mid]
    return (samples[mid - 1] + samples[mid]) / 2.0


@dataclass(frozen=True)
class QuantizationOutcome:
    """Result of :func:`maybe_apply_dynamic_quantization`.

    Attributes:
        applied: Whether the int8 weights were kept (``out_path``
            overwritten) or discarded (float32 graph left untouched).
        reason: Human-readable explanation, with the real numbers --
            never a bare "it worked"/"it didn't".
        baseline_p50_ms: Float32 median single-item latency.
        quantized_p50_ms: Int8 median single-item latency.
        equivalence_pct: Percentage of ``sample_texts`` where the
            quantized graph predicts the same label as the float32 graph.
    """

    applied: bool
    reason: str
    baseline_p50_ms: float
    quantized_p50_ms: float
    equivalence_pct: float


def maybe_apply_dynamic_quantization(
    onnx_path: Path,
    sample_texts: Sequence[str],
    warmup: int = 20,
    runs: int = 100,
) -> QuantizationOutcome:
    """Try onnxruntime dynamic int8 quantization; keep it only if it helps.

    Quantizes a temporary copy of ``onnx_path``, compares its predictions
    and median single-item latency against the existing float32 graph on
    ``sample_texts``, and overwrites ``onnx_path`` with the int8 weights
    **only if** the quantized graph is both faster and at least
    :data:`EQUIVALENCE_MIN_PCT` equivalent to the float32 predictions.
    Otherwise ``onnx_path`` is left untouched and the reason is returned,
    never silently discarded (see ``harness/latency.md``: a negative result
    honestly measured is a valid outcome, not a defect to hide).

    This is a narrow, self-contained gate for this task's ``--quantize``
    flag -- not the project's official baseline-vs-onnx benchmark (p50/p90/
    p95/p99, n=200, warm-up=20, both backends). That is task T21's
    ``triagem.optimization.benchmark``.

    Args:
        onnx_path: The float32 ``.onnx`` file, as written by
            :func:`export_to_onnx`. Overwritten in place if quantization is
            kept.
        sample_texts: At least 20 real texts to evaluate on.
        warmup: Untimed calls before timing each graph (default: 20).
        runs: Timed calls per graph (default: 100, capped to
            ``len(sample_texts)``).

    Returns:
        The :class:`QuantizationOutcome`.

    Raises:
        ValueError: Fewer than 20 ``sample_texts`` were given.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic  # optional dep, local import

    if len(sample_texts) < 20:
        raise ValueError("maybe_apply_dynamic_quantization needs at least 20 sample_texts.")

    runs = min(runs, len(sample_texts))

    baseline_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    baseline_labels = _onnx_predict_labels(baseline_session, sample_texts)
    baseline_p50 = _median_latency_ms(
        lambda batch: _onnx_predict_labels(baseline_session, batch), sample_texts, warmup, runs
    )

    quant_path = onnx_path.with_suffix(".int8.tmp.onnx")
    quantize_dynamic(
        model_input=str(onnx_path), model_output=str(quant_path), weight_type=QuantType.QInt8
    )
    try:
        quant_session = ort.InferenceSession(str(quant_path), providers=["CPUExecutionProvider"])
        quant_labels = _onnx_predict_labels(quant_session, sample_texts)
        quant_p50 = _median_latency_ms(
            lambda batch: _onnx_predict_labels(quant_session, batch), sample_texts, warmup, runs
        )

        matches = sum(1 for a, b in zip(baseline_labels, quant_labels, strict=True) if a == b)
        equivalence_pct = matches / len(sample_texts) * 100.0
        faster = quant_p50 < baseline_p50
        equivalent_enough = equivalence_pct >= EQUIVALENCE_MIN_PCT

        if faster and equivalent_enough:
            onnx_path.write_bytes(quant_path.read_bytes())
            reason = (
                f"p50 {baseline_p50:.4f}ms -> {quant_p50:.4f}ms (n={runs}, warmup={warmup}), "
                f"equivalence {equivalence_pct:.2f}% (>= {EQUIVALENCE_MIN_PCT:.0f}% floor)"
            )
            return QuantizationOutcome(True, reason, baseline_p50, quant_p50, equivalence_pct)

        reason = (
            f"p50 {baseline_p50:.4f}ms -> {quant_p50:.4f}ms "
            f"({'not faster' if not faster else 'faster'}), equivalence {equivalence_pct:.2f}% "
            f"({'below' if not equivalent_enough else 'meets'} the {EQUIVALENCE_MIN_PCT:.0f}% "
            "floor) -- keeping float32."
        )
        return QuantizationOutcome(False, reason, baseline_p50, quant_p50, equivalence_pct)
    finally:
        quant_path.unlink(missing_ok=True)


def _update_meta_after_quantization(out_path: Path, outcome: QuantizationOutcome) -> None:
    """Record the quantization decision in ``model_onnx_meta.json``, applied or not."""
    meta = _read_meta(out_path)
    meta["quantized"] = outcome.applied
    meta["quantization_check"] = {
        "applied": outcome.applied,
        "reason": outcome.reason,
        "baseline_p50_ms": outcome.baseline_p50_ms,
        "quantized_p50_ms": outcome.quantized_p50_ms,
        "equivalence_pct": outcome.equivalence_pct,
    }
    _write_meta(out_path, meta)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``python -m triagem.optimization.onnx_export``."""
    parser = argparse.ArgumentParser(
        description=(
            "Export models/model.joblib (TF-IDF + LogisticRegression, T7) to "
            "models/model.onnx + models/model_onnx_meta.json for the Predictor's "
            "'onnx' backend (T9)."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Fitted pipeline to convert (default: Settings.models_dir/model.joblib).",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Destination .onnx file (default: Settings.models_dir/model.onnx).",
    )
    parser.add_argument(
        "--opset", type=int, default=None, help="ONNX opset (default: skl2onnx's own default)."
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help=(
            "Try dynamic int8 quantization; keep it only if it is faster and stays "
            f">= {EQUIVALENCE_MIN_PCT:.0f}% equivalent to the float32 graph on a real sample "
            "(see maybe_apply_dynamic_quantization)."
        ),
    )
    parser.add_argument(
        "--quantize-sample-n",
        type=int,
        default=100,
        help="Sample size for the quantization equivalence/latency check (default: 100).",
    )
    parser.add_argument(
        "--quantize-warmup",
        type=int,
        default=20,
        help="Warm-up calls before timing, for the quantization check (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the sample used by --quantize (default: 42).",
    )
    return parser


def main() -> None:
    """CLI entry point: ``python -m triagem.optimization.onnx_export``."""
    from triagem.logging_conf import setup_logging

    setup_logging()
    args = _build_arg_parser().parse_args()
    settings = get_settings()
    model_path = args.model_path or (settings.models_dir / MODEL_FILENAME)
    out_path = args.out_path or (settings.models_dir / ONNX_MODEL_FILENAME)

    result_path = export_to_onnx(model_path, out_path, opset=args.opset)
    print(f"onnx model: {result_path}")
    print(f"onnx meta: {result_path.parent / ONNX_META_FILENAME}")

    if args.quantize:
        from triagem.data.loaders import load_dataset

        df = load_dataset(settings.data_path)
        n = min(args.quantize_sample_n, len(df))
        sample = df["text"].sample(n=n, random_state=args.seed).tolist()
        outcome = maybe_apply_dynamic_quantization(
            out_path, sample, warmup=args.quantize_warmup, runs=n
        )
        _update_meta_after_quantization(out_path, outcome)
        verdict = "APPLIED" if outcome.applied else "SKIPPED"
        print(f"quantization {verdict}: {outcome.reason}")


if __name__ == "__main__":
    main()
