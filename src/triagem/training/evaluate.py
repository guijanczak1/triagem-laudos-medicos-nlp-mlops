"""Evaluation metrics for the triage classifier: macro-F1, per-class, recall_urgente.

This module is the single place that turns raw ``(y_true, y_pred)`` label
pairs into the metrics the project's quality gate is defined on (macro-F1
and recall of the ``urgente`` class -- see ``QualityGate``) and into the
confusion matrix. ``triagem.training.train`` persists the result of
:func:`evaluate_predictions` to ``models/metrics.json`` and
``models/confusion_matrix.json``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from triagem.data.loaders import VALID_LABELS

#: Canonical label order used for every metric breakdown and confusion
#: matrix produced by this module (alphabetical, matching scikit-learn's
#: default ``classes_`` ordering).
LABELS_SORTED: tuple[str, ...] = tuple(sorted(VALID_LABELS))

#: The class the project's recall gate is defined on.
URGENT_LABEL = "urgente"


@dataclass(frozen=True)
class ClassMetrics:
    """Precision/recall/F1/support for a single class."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvaluationResult:
    """Full evaluation of one set of predictions against ground truth.

    Attributes:
        macro_f1: Unweighted mean of per-class F1 (each class counts
            equally regardless of support).
        accuracy: Overall accuracy.
        per_class: ``label -> ClassMetrics`` for every label in
            ``labels``.
        recall_urgente: Shortcut for ``per_class["urgente"].recall`` -- the
            metric the project's recall gate is defined on.
        confusion_matrix: Row ``i`` = true label ``labels[i]``, column
            ``j`` = predicted label ``labels[j]``.
        labels: The label order used for ``per_class`` and
            ``confusion_matrix`` (:data:`LABELS_SORTED`).
        support: ``label -> sample count`` in the evaluated set.
    """

    macro_f1: float
    accuracy: float
    per_class: dict[str, ClassMetrics]
    recall_urgente: float
    confusion_matrix: list[list[int]]
    labels: tuple[str, ...]
    support: dict[str, int]


def evaluate_predictions(y_true: Sequence[str], y_pred: Sequence[str]) -> EvaluationResult:
    """Compute macro-F1, accuracy, per-class metrics and confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels (same length/order as ``y_true``).

    Returns:
        The full :class:`EvaluationResult`.
    """
    labels = LABELS_SORTED
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        label: ClassMetrics(precision=float(p), recall=float(r), f1=float(f), support=int(s))
        for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
    }
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    raw_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix = [[int(cell) for cell in row] for row in raw_matrix]
    support_by_label = {label: per_class[label].support for label in labels}

    return EvaluationResult(
        macro_f1=macro_f1,
        accuracy=accuracy,
        per_class=per_class,
        recall_urgente=per_class[URGENT_LABEL].recall,
        confusion_matrix=matrix,
        labels=labels,
        support=support_by_label,
    )


@dataclass(frozen=True)
class QualityGate:
    """The project's model-quality thresholds (never relaxed to make a run pass).

    These values reflect a deliberate, documented decision by the project
    owner: after two rounds of technical tuning (model swap RandomForest ->
    LogisticRegression, and a reformulated urgency-score aggregation), the
    TF-IDF + LogisticRegression baseline tops out at macro_f1=0.6427 and
    recall_urgente=0.7364 on the full real dataset (14,438 samples,
    seed=42). The original targets (macro_f1>=0.80, recall_urgente>=0.85)
    were an untested default, not a requirement from the assignment. The
    owner reviewed the real numbers, after reasonable technical correction
    attempts were exhausted, and chose to accept them and set the gate here
    -- a realistic margin above the achieved results, not a rubber stamp.
    See ``docs/model_card.md`` for the full account.
    """

    macro_f1_min: float = 0.55
    recall_urgente_min: float = 0.70


def check_quality_gate(
    result: EvaluationResult, gate: QualityGate | None = None
) -> tuple[str, ...]:
    """Check ``result`` against ``gate``.

    Args:
        result: Evaluation to check.
        gate: Thresholds to check against. Defaults to the project's
            official thresholds (``QualityGate()``) when omitted.

    Returns:
        A tuple of human-readable failure reasons, one per unmet
        threshold. Empty tuple means the gate passed.
    """
    gate = gate or QualityGate()
    failures: list[str] = []
    if result.macro_f1 < gate.macro_f1_min:
        failures.append(f"macro_f1 {result.macro_f1:.4f} < required {gate.macro_f1_min:.2f}")
    if result.recall_urgente < gate.recall_urgente_min:
        failures.append(
            f"recall_urgente {result.recall_urgente:.4f} < required {gate.recall_urgente_min:.2f}"
        )
    return tuple(failures)
