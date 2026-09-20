"""Tests for triagem.training.train.

Trains on the small, deterministic ``tests/fixtures/laudos_sample.csv``
fixture (54 real abstracts, all 3 labels, no network) and checks: the four
artifacts are written with the right shape/content, a saved-artifact
inference round-trip, seed-determinism (tolerance 1e-9), that the quality
gate is reported honestly (never silently forced to pass) and the CLI
entry point.

This is a sanity/contract suite, not the project's real quality gate --
per ``harness/sklearn.md``'s Definition of Done, the gate itself (macro-F1
>= 0.80, recall_urgente >= 0.85) is checked by actually running
``python -m triagem.training.train`` against the real, full processed
dataset, not this 54-row fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

from triagem.config import Settings
from triagem.data.loaders import VALID_LABELS
from triagem.training.evaluate import QualityGate
from triagem.training.train import (
    CONFUSION_MATRIX_FILENAME,
    LABEL_ENCODER_FILENAME,
    METRICS_FILENAME,
    MODEL_FILENAME,
    TrainConfig,
    TrainResult,
    main,
    train,
)

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "laudos_sample.csv"


@pytest.fixture
def cfg(tmp_path: Path) -> TrainConfig:
    """A TrainConfig pointed at the small fixture and a scratch models_dir."""
    return TrainConfig(data_path=FIXTURE_PATH, models_dir=tmp_path / "models", seed=42)


class TestTrainArtifacts:
    """train() must write exactly the four deliverable artifacts."""

    def test_writes_all_four_artifact_files(self, cfg: TrainConfig) -> None:
        result = train(cfg)

        assert result.model_path.exists()
        assert result.metrics_path.exists()
        assert result.label_encoder_path.exists()
        assert result.confusion_matrix_path.exists()
        assert result.model_path.name == MODEL_FILENAME
        assert result.metrics_path.name == METRICS_FILENAME
        assert result.label_encoder_path.name == LABEL_ENCODER_FILENAME
        assert result.confusion_matrix_path.name == CONFUSION_MATRIX_FILENAME

    def test_returns_train_result_dataclass(self, cfg: TrainConfig) -> None:
        result = train(cfg)

        assert isinstance(result, TrainResult)
        assert isinstance(result.macro_f1, float)
        assert isinstance(result.recall_urgente, float)
        assert result.n_samples > 0

    def test_creates_models_dir_if_missing(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "does" / "not" / "exist"
        cfg = TrainConfig(data_path=FIXTURE_PATH, models_dir=models_dir, seed=42)

        train(cfg)

        assert models_dir.exists()


class TestMetricsJsonContent:
    """metrics.json must expose every field the acceptance criteria list."""

    def test_metrics_json_has_all_required_top_level_fields(self, cfg: TrainConfig) -> None:
        result = train(cfg)

        payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))

        for key in (
            "macro_f1",
            "accuracy",
            "per_class",
            "recall_urgente",
            "support",
            "seed",
            "n_samples",
            "trained_at",
            "sklearn_version",
        ):
            assert key in payload, f"metrics.json missing required field {key!r}"

    def test_per_class_has_precision_recall_f1_for_every_label(self, cfg: TrainConfig) -> None:
        result = train(cfg)
        payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))

        assert set(payload["per_class"].keys()) == set(VALID_LABELS)
        for label_metrics in payload["per_class"].values():
            assert set(label_metrics.keys()) == {"precision", "recall", "f1", "support"}

    def test_seed_and_n_samples_match_the_run(self, cfg: TrainConfig) -> None:
        result = train(cfg)
        payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))

        assert payload["seed"] == 42
        assert payload["n_samples"] == result.n_samples

    def test_no_absolute_machine_path_is_written_to_metrics(
        self, cfg: TrainConfig, tmp_path: Path
    ) -> None:
        result = train(cfg)
        raw_text = result.metrics_path.read_text(encoding="utf-8")

        assert str(tmp_path) not in raw_text
        assert str(FIXTURE_PATH) not in raw_text

    def test_label_encoder_json_lists_the_fitted_classes(self, cfg: TrainConfig) -> None:
        result = train(cfg)
        payload = json.loads(result.label_encoder_path.read_text(encoding="utf-8"))

        assert set(payload["classes"]) == set(VALID_LABELS)

    def test_confusion_matrix_json_is_square_and_labeled(self, cfg: TrainConfig) -> None:
        result = train(cfg)
        payload = json.loads(result.confusion_matrix_path.read_text(encoding="utf-8"))

        labels = payload["labels"]
        matrix = payload["matrix"]
        assert len(labels) == 3
        assert len(matrix) == 3
        assert all(len(row) == 3 for row in matrix)


class TestInferenceFromSavedArtifact:
    """Loading model.joblib from disk must predict without error (DoD requirement)."""

    def test_loaded_model_predicts_with_correct_shape_and_label_domain(
        self, cfg: TrainConfig
    ) -> None:
        result = train(cfg)

        loaded = joblib.load(result.model_path)
        texts = ["Acute severe emergency with critical organ failure.", "Routine mild follow-up."]

        predictions = loaded.predict(texts)

        assert len(predictions) == len(texts)
        assert set(predictions).issubset(set(VALID_LABELS))

    def test_loaded_model_predict_proba_sums_to_one(self, cfg: TrainConfig) -> None:
        result = train(cfg)
        loaded = joblib.load(result.model_path)

        proba = loaded.predict_proba(["Chronic stable benign condition."])

        assert proba.shape == (1, 3)
        assert proba.sum() == pytest.approx(1.0, abs=1e-6)


class TestReproducibility:
    """Same seed + same data => same metrics (tolerance 1e-9)."""

    def test_two_runs_with_the_same_seed_produce_identical_metrics(self, tmp_path: Path) -> None:
        cfg_a = TrainConfig(data_path=FIXTURE_PATH, models_dir=tmp_path / "run_a", seed=42)
        cfg_b = TrainConfig(data_path=FIXTURE_PATH, models_dir=tmp_path / "run_b", seed=42)

        result_a = train(cfg_a)
        result_b = train(cfg_b)

        assert result_a.macro_f1 == pytest.approx(result_b.macro_f1, abs=1e-9)
        assert result_a.recall_urgente == pytest.approx(result_b.recall_urgente, abs=1e-9)

    def test_different_seeds_may_produce_different_metrics_without_erroring(
        self, tmp_path: Path
    ) -> None:
        cfg_a = TrainConfig(data_path=FIXTURE_PATH, models_dir=tmp_path / "run_a", seed=1)
        cfg_b = TrainConfig(data_path=FIXTURE_PATH, models_dir=tmp_path / "run_b", seed=2)

        result_a = train(cfg_a)
        result_b = train(cfg_b)

        # No assertion on equality/inequality of the metrics themselves --
        # only that different seeds produce a different stratified split
        # deterministically and neither run errors.
        assert isinstance(result_a.macro_f1, float)
        assert isinstance(result_b.macro_f1, float)


class TestQualityGateReporting:
    """The gate result must reflect the real numbers, never be forced to pass."""

    def test_gate_passed_true_when_thresholds_trivially_met(self, tmp_path: Path) -> None:
        cfg = TrainConfig(
            data_path=FIXTURE_PATH,
            models_dir=tmp_path / "models",
            seed=42,
            quality_gate=QualityGate(macro_f1_min=0.0, recall_urgente_min=0.0),
        )

        result = train(cfg)

        assert result.gate_passed is True
        assert result.gate_failures == ()

    def test_gate_passed_false_and_real_numbers_reported_when_impossible_to_meet(
        self, tmp_path: Path
    ) -> None:
        cfg = TrainConfig(
            data_path=FIXTURE_PATH,
            models_dir=tmp_path / "models",
            seed=42,
            quality_gate=QualityGate(macro_f1_min=1.01, recall_urgente_min=1.01),
        )

        result = train(cfg)

        assert result.gate_passed is False
        assert len(result.gate_failures) == 2
        assert f"{result.macro_f1:.4f}" in "".join(result.gate_failures)
        assert f"{result.recall_urgente:.4f}" in "".join(result.gate_failures)

    def test_metrics_json_records_gate_outcome(self, tmp_path: Path) -> None:
        cfg = TrainConfig(
            data_path=FIXTURE_PATH,
            models_dir=tmp_path / "models",
            seed=42,
            quality_gate=QualityGate(macro_f1_min=1.01, recall_urgente_min=1.01),
        )

        result = train(cfg)
        payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))

        assert payload["gate_passed"] is False
        assert len(payload["gate_failures"]) == 2


class TestCliMain:
    """python -m triagem.training.train"""

    def test_cli_runs_end_to_end_and_prints_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Artifacts + metrics are produced and printed, whatever the gate outcome.

        The 54-row fixture is far too small/homogeneous to reliably clear the
        project's real quality gate (that is checked on the full dataset, not
        here -- see the module docstring), so this only asserts the CLI
        mechanics: it must run to completion (pass) or exit(1) (gate failed),
        never crash, and it must always write the artifacts and print the
        real numbers either way.
        """
        models_dir = tmp_path / "models"
        monkeypatch.setattr("triagem.training.train.get_settings", lambda: Settings())
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--data-path",
                str(FIXTURE_PATH),
                "--models-dir",
                str(models_dir),
                "--seed",
                "42",
            ],
        )

        try:
            main()
        except SystemExit as exc:
            assert exc.code == 1

        captured = capsys.readouterr()
        assert "macro_f1=" in captured.out
        assert "recall_urgente=" in captured.out
        assert (models_dir / MODEL_FILENAME).exists()

    def test_cli_exits_non_zero_when_quality_gate_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """main() must propagate a failing gate as a non-zero exit, never mask it."""
        failing_result = TrainResult(
            model_path=tmp_path / "model.joblib",
            metrics_path=tmp_path / "metrics.json",
            label_encoder_path=tmp_path / "label_encoder.json",
            confusion_matrix_path=tmp_path / "confusion_matrix.json",
            macro_f1=0.42,
            recall_urgente=0.10,
            n_samples=100,
            gate_passed=False,
            gate_failures=(
                "macro_f1 0.4200 < required 0.80",
                "recall_urgente 0.1000 < required 0.85",
            ),
        )
        monkeypatch.setattr("triagem.training.train.train", lambda cfg: failing_result)
        monkeypatch.setattr("sys.argv", ["prog"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "QUALITY GATE FAILED" in captured.out
        assert "0.4200" in captured.out
