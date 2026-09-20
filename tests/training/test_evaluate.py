"""Tests for triagem.training.evaluate.

Uses small, hand-built ``y_true``/``y_pred`` lists (no model, no fixture
CSV) so the metric math itself -- macro-F1, per-class breakdown,
``recall_urgente``, confusion matrix and the quality gate check -- is
verified independently of the pipeline.
"""

from __future__ import annotations

from triagem.training.evaluate import (
    LABELS_SORTED,
    QualityGate,
    check_quality_gate,
    evaluate_predictions,
)


class TestEvaluatePredictionsPerfect:
    """All predictions correct: every metric should be at its ceiling."""

    def test_perfect_predictions_score_one_everywhere(self) -> None:
        y_true = ["normal", "atencao", "urgente"] * 4
        y_pred = list(y_true)

        result = evaluate_predictions(y_true, y_pred)

        assert result.macro_f1 == 1.0
        assert result.accuracy == 1.0
        assert result.recall_urgente == 1.0
        for label in LABELS_SORTED:
            metrics = result.per_class[label]
            assert metrics.precision == 1.0
            assert metrics.recall == 1.0
            assert metrics.f1 == 1.0

    def test_confusion_matrix_is_diagonal_for_perfect_predictions(self) -> None:
        y_true = ["normal", "atencao", "urgente"] * 3
        y_pred = list(y_true)

        result = evaluate_predictions(y_true, y_pred)

        assert result.labels == LABELS_SORTED
        for i, row in enumerate(result.confusion_matrix):
            for j, cell in enumerate(row):
                assert cell == (3 if i == j else 0)


class TestEvaluatePredictionsImperfect:
    """Known confusion pattern: metrics must match hand-computed values."""

    def test_urgente_missed_entirely_gives_zero_recall_urgente(self) -> None:
        # Every "urgente" sample is (wrongly) predicted as "normal".
        y_true = ["normal", "normal", "urgente", "urgente", "atencao", "atencao"]
        y_pred = ["normal", "normal", "normal", "normal", "atencao", "atencao"]

        result = evaluate_predictions(y_true, y_pred)

        assert result.recall_urgente == 0.0
        assert result.per_class["urgente"].support == 2
        assert result.per_class["normal"].precision == 0.5  # 2 correct out of 4 predicted normal

    def test_support_reflects_true_label_counts(self) -> None:
        y_true = ["normal", "normal", "normal", "urgente", "atencao"]
        y_pred = ["normal", "normal", "urgente", "urgente", "atencao"]

        result = evaluate_predictions(y_true, y_pred)

        assert result.support == {"atencao": 1, "normal": 3, "urgente": 1}


class TestCheckQualityGate:
    """check_quality_gate(): returns real failure strings, never silently passes."""

    def test_gate_passes_when_both_thresholds_met(self) -> None:
        y_true = ["normal", "atencao", "urgente"] * 4
        result = evaluate_predictions(y_true, y_true)
        gate = QualityGate()

        failures = check_quality_gate(result, gate)

        assert failures == ()

    def test_gate_reports_macro_f1_failure_with_real_number(self) -> None:
        y_true = ["normal"] * 5 + ["urgente"] * 5
        y_pred = ["normal"] * 10  # urgente never predicted -> low macro-F1
        result = evaluate_predictions(y_true, y_pred)
        gate = QualityGate()

        failures = check_quality_gate(result, gate)

        assert any("macro_f1" in f for f in failures)
        assert f"{result.macro_f1:.4f}" in "".join(failures)

    def test_gate_reports_recall_urgente_failure_with_real_number(self) -> None:
        y_true = ["urgente"] * 10
        y_pred = ["normal"] * 5 + ["urgente"] * 5  # recall_urgente = 0.5
        result = evaluate_predictions(y_true, y_pred)
        gate = QualityGate(macro_f1_min=0.0, recall_urgente_min=QualityGate().recall_urgente_min)

        failures = check_quality_gate(result, gate)

        assert len(failures) == 1
        assert "recall_urgente" in failures[0]
        assert "0.5000" in failures[0]

    def test_default_gate_uses_project_official_thresholds(self) -> None:
        gate = QualityGate()

        assert gate.macro_f1_min == 0.55
        assert gate.recall_urgente_min == 0.70
