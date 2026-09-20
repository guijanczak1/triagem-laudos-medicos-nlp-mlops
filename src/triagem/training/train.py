"""Train the TF-IDF + LogisticRegression baseline and persist versionable artifacts.

``train()`` is the single entry point that turns the processed dataset
(``triagem.data.loaders.load_dataset``, T4/T5) into a fitted pipeline
(``triagem.models.pipeline.build_pipeline``, T6) and four artifacts under
``models/``:

- ``model.joblib`` -- the fitted ``sklearn.pipeline.Pipeline``.
- ``metrics.json`` -- macro-F1, accuracy, per-class precision/recall/F1,
  ``recall_urgente`` and run metadata (see ``triagem.training.evaluate``).
- ``label_encoder.json`` -- the class order the fitted classifier predicts
  in (``predict_proba`` columns / ``classes_``).
- ``confusion_matrix.json`` -- the holdout confusion matrix.

The project's quality gate (macro-F1 >= 0.55, recall of ``urgente`` >= 0.70
-- see ``QualityGate`` in ``triagem.training.evaluate``) is checked but
never used to reject or rewrite the metrics: ``train()`` always writes the
real numbers and reports ``gate_passed``/``gate_failures`` alongside them,
so a run that misses the target is fully visible rather than silently
accepted. These thresholds were consciously lowered from the original
0.80/0.85 targets by the project owner after real tuning topped out at
macro_f1=0.6427/recall_urgente=0.7364 -- see ``docs/model_card.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import sklearn

from triagem.config import Settings, get_settings
from triagem.data.loaders import load_dataset, split_dataset
from triagem.models.pipeline import CLASSIFIER_STEP, build_pipeline
from triagem.training.evaluate import QualityGate, check_quality_gate, evaluate_predictions

logger = logging.getLogger(__name__)

MODEL_FILENAME = "model.joblib"
METRICS_FILENAME = "metrics.json"
LABEL_ENCODER_FILENAME = "label_encoder.json"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.json"


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for one training run.

    Attributes:
        data_path: Processed dataset CSV. Defaults to ``Settings.data_path``.
        models_dir: Destination directory for the four artifacts. Defaults
            to ``Settings.models_dir``.
        seed: Random seed forwarded to the split and to the pipeline's
            ``LogisticRegression``. Same seed + same data => same
            metrics (tolerance 1e-9).
        max_features: TF-IDF vocabulary cap, forwarded to
            ``build_pipeline``.
        test_size: Fraction of the dataset reserved as the stratified
            holdout the reported metrics are computed on.
        classifier_c: Optional override of the pipeline's
            ``LogisticRegression(C=...)`` inverse-regularization strength,
            applied via ``Pipeline.set_params`` so
            ``triagem.models.pipeline`` (T6, skl2onnx-exportable by
            contract) is never edited just to try a different
            regularization strength. ``None`` keeps T6's default (0.1).
        quality_gate: Thresholds to check the holdout metrics against.
            Defaults to the project's official thresholds (macro_f1>=0.55,
            recall_urgente>=0.70, consciously lowered by the project owner
            from the original 0.80/0.85 targets -- see
            ``docs/model_card.md``) -- never relax this further to make a
            run "pass".
    """

    data_path: Path | None = None
    models_dir: Path | None = None
    seed: int = 42
    max_features: int = 20000
    test_size: float = 0.2
    classifier_c: float | None = None
    quality_gate: QualityGate = field(default_factory=QualityGate)


@dataclass(frozen=True)
class TrainResult:
    """Outcome of one ``train()`` call: artifact locations + headline metrics."""

    model_path: Path
    metrics_path: Path
    label_encoder_path: Path
    confusion_matrix_path: Path
    macro_f1: float
    recall_urgente: float
    n_samples: int
    gate_passed: bool
    gate_failures: tuple[str, ...]


def train(cfg: TrainConfig | None = None) -> TrainResult:
    """Train, evaluate on a stratified holdout, and persist all artifacts.

    Args:
        cfg: Training configuration. Defaults to ``TrainConfig()``.

    Returns:
        A :class:`TrainResult` with the real holdout metrics and whether
        they meet ``cfg.quality_gate`` -- always populated with the actual
        numbers, whether the gate passed or not.

    Raises:
        DatasetSchemaError: Propagated from ``load_dataset`` if the
            processed dataset is missing or malformed.
    """
    cfg = cfg or TrainConfig()
    settings: Settings = get_settings()
    data_path = cfg.data_path if cfg.data_path is not None else settings.data_path
    models_dir = cfg.models_dir if cfg.models_dir is not None else settings.models_dir

    df = load_dataset(data_path)
    train_df, test_df = split_dataset(df, test_size=cfg.test_size, seed=cfg.seed)

    pipeline = build_pipeline(seed=cfg.seed, max_features=cfg.max_features)
    if cfg.classifier_c is not None:
        pipeline.set_params(**{f"{CLASSIFIER_STEP}__C": cfg.classifier_c})

    pipeline.fit(train_df["text"].tolist(), train_df["label"].tolist())
    predictions = pipeline.predict(test_df["text"].tolist())

    result = evaluate_predictions(test_df["label"].tolist(), list(predictions))
    gate_failures = check_quality_gate(result, cfg.quality_gate)
    gate_passed = len(gate_failures) == 0

    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / MODEL_FILENAME
    joblib.dump(pipeline, model_path)

    classifier = pipeline.named_steps[CLASSIFIER_STEP]
    label_encoder_path = models_dir / LABEL_ENCODER_FILENAME
    _write_json(
        label_encoder_path,
        {"classes": [str(label) for label in classifier.classes_]},
    )

    confusion_matrix_path = models_dir / CONFUSION_MATRIX_FILENAME
    _write_json(
        confusion_matrix_path,
        {"labels": list(result.labels), "matrix": result.confusion_matrix},
    )

    metrics_path = models_dir / METRICS_FILENAME
    _write_json(
        metrics_path,
        {
            "macro_f1": result.macro_f1,
            "accuracy": result.accuracy,
            "per_class": {
                label: {
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "support": metrics.support,
                }
                for label, metrics in result.per_class.items()
            },
            "recall_urgente": result.recall_urgente,
            "support": result.support,
            "seed": cfg.seed,
            "n_samples": int(len(df)),
            "trained_at": datetime.now(UTC).isoformat(),
            "sklearn_version": sklearn.__version__,
            "gate_passed": gate_passed,
            "gate_failures": list(gate_failures),
        },
    )

    if gate_passed:
        logger.info(
            "train complete: macro_f1=%.4f recall_urgente=%.4f gate=PASSED",
            result.macro_f1,
            result.recall_urgente,
        )
    else:
        logger.warning(
            "train complete: macro_f1=%.4f recall_urgente=%.4f gate=FAILED (%s)",
            result.macro_f1,
            result.recall_urgente,
            "; ".join(gate_failures),
        )

    return TrainResult(
        model_path=model_path,
        metrics_path=metrics_path,
        label_encoder_path=label_encoder_path,
        confusion_matrix_path=confusion_matrix_path,
        macro_f1=result.macro_f1,
        recall_urgente=result.recall_urgente,
        n_samples=int(len(df)),
        gate_passed=gate_passed,
        gate_failures=gate_failures,
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write ``payload`` as pretty, sorted-key JSON. No machine-specific paths."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``python -m triagem.training.train``."""
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the TF-IDF + LogisticRegression triage baseline, "
            "writing models/model.joblib, metrics.json, label_encoder.json "
            "and confusion_matrix.json."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Processed dataset CSV (defaults to Settings.data_path).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Destination directory for artifacts (defaults to Settings.models_dir).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--max-features",
        type=int,
        default=20000,
        help="TF-IDF vocabulary cap (default: 20000).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset held out for evaluation (default: 0.2).",
    )
    parser.add_argument(
        "--classifier-c",
        type=float,
        default=None,
        help="Override the LogisticRegression's C, inverse regularization (default: T6's 0.1).",
    )
    return parser


def main() -> None:
    """CLI entry point: ``python -m triagem.training.train --seed 42``.

    Exits with a non-zero status (and prints the unmet thresholds) if the
    trained model does not meet the project's quality gate -- the
    thresholds are never adjusted to force a pass.
    """
    from triagem.logging_conf import setup_logging

    setup_logging()
    args = _build_arg_parser().parse_args()
    cfg = TrainConfig(
        data_path=args.data_path,
        models_dir=args.models_dir,
        seed=args.seed,
        max_features=args.max_features,
        test_size=args.test_size,
        classifier_c=args.classifier_c,
    )
    result = train(cfg)

    print(f"model: {result.model_path}")
    print(f"metrics: {result.metrics_path}")
    print(
        f"n_samples={result.n_samples} macro_f1={result.macro_f1:.4f} "
        f"recall_urgente={result.recall_urgente:.4f}"
    )

    if not result.gate_passed:
        print("QUALITY GATE FAILED: " + "; ".join(result.gate_failures))
        raise SystemExit(1)

    print("QUALITY GATE PASSED")


if __name__ == "__main__":
    main()
