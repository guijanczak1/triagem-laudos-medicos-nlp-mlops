"""Tests for triagem.optimization.onnx_export.

Fits the real T6 pipeline on the small, deterministic
``tests/fixtures/laudos_sample.csv`` fixture, exports it to ONNX with
:func:`export_to_onnx`, and validates -- against the real ``onnxruntime``
runtime and the real :class:`~triagem.serving.predictor.Predictor` public
contract, never just "the export did not raise" -- that:

- the ``.onnx`` file + ``model_onnx_meta.json`` sidecar are written with the
  documented shape;
- ``Predictor.load(backend="onnx")`` on the exported artifact predicts with
  the exact same ``Prediction`` contract as the ``sklearn`` backend;
- sklearn and ONNX predictions agree on (almost) every fixture text;
- ``maybe_apply_dynamic_quantization`` never corrupts ``model.onnx``,
  whatever it decides.

The much larger, real-dataset (2,888-row test split) sklearn-vs-onnx
equivalence comparison for this task's report was run separately (not as an
automated test -- that full comparison, with p50/p90/p95/p99, is task T21's
``tests/optimization/test_equivalence.py`` deliverable); see the task's
NOTES for the real number (99.52%).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import onnxruntime as ort
import pytest
from sklearn.pipeline import Pipeline

from triagem.config import Settings
from triagem.data.loaders import VALID_LABELS, load_dataset
from triagem.exceptions import ModelArtifactNotFound
from triagem.models.pipeline import CLASSIFIER_STEP, build_pipeline
from triagem.optimization import onnx_export
from triagem.optimization.onnx_export import (
    ONNX_META_FILENAME,
    ONNX_MODEL_FILENAME,
    QuantizationOutcome,
    export_to_onnx,
    maybe_apply_dynamic_quantization,
)
from triagem.serving.predictor import Predictor
from triagem.training.train import LABEL_ENCODER_FILENAME, MODEL_FILENAME

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "laudos_sample.csv"


@pytest.fixture(scope="module")
def fitted_pipeline() -> Pipeline:
    """Fit the real T6 pipeline on the small fixture once per test module."""
    df = load_dataset(FIXTURE_PATH)
    pipeline = build_pipeline(seed=42)
    pipeline.fit(df["text"].tolist(), df["label"].tolist())
    return pipeline


@pytest.fixture
def models_dir_with_sklearn_artifact(tmp_path: Path, fitted_pipeline: Pipeline) -> Path:
    """A models/ dir with a real joblib artifact, ready to export."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(fitted_pipeline, models_dir / MODEL_FILENAME)
    return models_dir


@pytest.fixture(autouse=True)
def _clear_predictor_cache() -> object:
    """Every test gets an empty Predictor cache, and leaves none behind."""
    Predictor.clear_cache()
    yield
    Predictor.clear_cache()


class TestExportToOnnx:
    """export_to_onnx() writes a working model.onnx + meta sidecar."""

    def test_raises_model_artifact_not_found_when_model_joblib_missing(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ModelArtifactNotFound) as exc_info:
            export_to_onnx(tmp_path / "missing.joblib", tmp_path / "model.onnx")

        assert "missing.joblib" in str(exc_info.value)

    def test_writes_onnx_file_and_meta_sidecar(
        self, models_dir_with_sklearn_artifact: Path
    ) -> None:
        out_path = models_dir_with_sklearn_artifact / ONNX_MODEL_FILENAME

        result = export_to_onnx(models_dir_with_sklearn_artifact / MODEL_FILENAME, out_path)

        assert result == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        meta_path = models_dir_with_sklearn_artifact / ONNX_META_FILENAME
        assert meta_path.exists()

    def test_meta_sidecar_has_documented_fields(
        self, models_dir_with_sklearn_artifact: Path
    ) -> None:
        out_path = models_dir_with_sklearn_artifact / ONNX_MODEL_FILENAME
        export_to_onnx(models_dir_with_sklearn_artifact / MODEL_FILENAME, out_path)

        meta = json.loads((models_dir_with_sklearn_artifact / ONNX_META_FILENAME).read_text())

        assert set(meta["classes"]) == set(VALID_LABELS)
        assert meta["opset"] is not None and meta["opset"] > 0
        assert meta["sklearn_version"]
        assert meta["skl2onnx_version"]
        assert meta["onnx_version"]
        assert meta["onnxruntime_version"]
        assert meta["quantized"] is False

    def test_strip_accents_adjustment_is_recorded_transparently(
        self, models_dir_with_sklearn_artifact: Path
    ) -> None:
        """T6's TfidfVectorizer(strip_accents='unicode') is not skl2onnx-exportable
        as-is; export_to_onnx must adjust a throwaway copy and say so in the meta
        file, never silently."""
        out_path = models_dir_with_sklearn_artifact / ONNX_MODEL_FILENAME
        export_to_onnx(models_dir_with_sklearn_artifact / MODEL_FILENAME, out_path)

        meta = json.loads((models_dir_with_sklearn_artifact / ONNX_META_FILENAME).read_text())
        assert meta["strip_accents_adjusted_for_export"] is True

    def test_does_not_mutate_the_original_fitted_pipeline_object(
        self, models_dir_with_sklearn_artifact: Path, fitted_pipeline: Pipeline
    ) -> None:
        """The in-memory pipeline (and therefore model.joblib on disk) used by the
        sklearn backend must never be touched by the export."""
        vectorizer = fitted_pipeline.named_steps["tfidf"]
        original_strip_accents = vectorizer.strip_accents

        export_to_onnx(
            models_dir_with_sklearn_artifact / MODEL_FILENAME,
            models_dir_with_sklearn_artifact / ONNX_MODEL_FILENAME,
        )

        assert vectorizer.strip_accents == original_strip_accents

    def test_exported_graph_loads_in_onnxruntime_with_expected_io(
        self, models_dir_with_sklearn_artifact: Path
    ) -> None:
        out_path = models_dir_with_sklearn_artifact / ONNX_MODEL_FILENAME
        export_to_onnx(models_dir_with_sklearn_artifact / MODEL_FILENAME, out_path)

        session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

        assert len(session.get_inputs()) == 1
        assert len(session.get_outputs()) == 2  # output_label, output_probability


class TestSklearnOnnxEquivalenceOnFixture:
    """sklearn and the exported ONNX graph must (almost always) agree."""

    def test_onnx_predictions_match_sklearn_on_the_fixture(
        self, models_dir_with_sklearn_artifact: Path, fitted_pipeline: Pipeline
    ) -> None:
        out_path = models_dir_with_sklearn_artifact / ONNX_MODEL_FILENAME
        export_to_onnx(models_dir_with_sklearn_artifact / MODEL_FILENAME, out_path)

        df = load_dataset(FIXTURE_PATH)
        texts = df["text"].tolist()

        sklearn_labels = [str(label) for label in fitted_pipeline.predict(texts)]
        session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        onnx_labels = onnx_export._onnx_predict_labels(session, texts)

        matches = sum(1 for a, b in zip(sklearn_labels, onnx_labels, strict=True) if a == b)
        equivalence_pct = matches / len(texts) * 100.0

        # A tiny (54-row) fixture can tip an occasional near-tied score across
        # the float64/float32 boundary; the real-dataset check (2,888 rows,
        # see this task's report) is the one held to the project's 99% floor.
        assert equivalence_pct >= 90.0, f"only {equivalence_pct:.1f}% equivalence on fixture"


class TestPredictorOnnxBackendEndToEnd:
    """The public Predictor contract must work for real on backend='onnx'.

    This is the actual T9 dependency this task closes: before T20 ran once,
    Predictor.load(backend='onnx') could only be tested against
    ModelArtifactNotFound (see tests/serving/test_predictor.py). Now that a
    real model.onnx exists, these tests exercise the real onnxruntime
    scoring path through the same public API the FastAPI service (T10) and
    benchmark (T21) use.
    """

    @pytest.fixture
    def onnx_models_dir(
        self, models_dir_with_sklearn_artifact: Path, fitted_pipeline: Pipeline
    ) -> Path:
        """A models/ dir with both a real model.joblib and a real model.onnx,
        plus the label_encoder.json sidecar the onnx backend falls back to."""
        classifier = fitted_pipeline.named_steps[CLASSIFIER_STEP]
        (models_dir_with_sklearn_artifact / LABEL_ENCODER_FILENAME).write_text(
            json.dumps({"classes": [str(c) for c in classifier.classes_]}), encoding="utf-8"
        )
        export_to_onnx(
            models_dir_with_sklearn_artifact / MODEL_FILENAME,
            models_dir_with_sklearn_artifact / ONNX_MODEL_FILENAME,
        )
        return models_dir_with_sklearn_artifact

    def test_predictor_loads_onnx_backend_without_error(self, onnx_models_dir: Path) -> None:
        predictor = Predictor.load(backend="onnx", models_dir=onnx_models_dir)

        assert predictor.metadata.backend == "onnx"
        assert set(predictor.metadata.classes) == set(VALID_LABELS)

    def test_predictor_onnx_predict_respects_the_prediction_contract(
        self, onnx_models_dir: Path
    ) -> None:
        predictor = Predictor.load(backend="onnx", models_dir=onnx_models_dir)

        result = predictor.predict(
            "Acute severe emergency with critical organ failure and hemorrhage."
        )

        assert result.backend == "onnx"
        assert set(result.scores) == set(VALID_LABELS)
        assert result.label in VALID_LABELS
        assert sum(result.scores.values()) == pytest.approx(1.0, abs=1e-6)
        assert result.latency_ms >= 0.0

    def test_predictor_onnx_predict_batch_matches_input_length_and_order(
        self, onnx_models_dir: Path
    ) -> None:
        predictor = Predictor.load(backend="onnx", models_dir=onnx_models_dir)
        texts = load_dataset(FIXTURE_PATH)["text"].tolist()

        results = predictor.predict_batch(texts)

        assert len(results) == len(texts)
        for result in results:
            assert result.backend == "onnx"
            assert sum(result.scores.values()) == pytest.approx(1.0, abs=1e-6)

    def test_sklearn_and_onnx_predictors_agree_on_the_fixture(self, onnx_models_dir: Path) -> None:
        """Cross-backend equivalence through the real public Predictor API
        (not just raw onnxruntime), sklearn Predictor vs onnx Predictor."""
        sklearn_predictor = Predictor.load(backend="sklearn", models_dir=onnx_models_dir)
        onnx_predictor = Predictor.load(backend="onnx", models_dir=onnx_models_dir)
        texts = load_dataset(FIXTURE_PATH)["text"].tolist()

        sklearn_labels = [p.label for p in sklearn_predictor.predict_batch(texts)]
        onnx_labels = [p.label for p in onnx_predictor.predict_batch(texts)]

        matches = sum(1 for a, b in zip(sklearn_labels, onnx_labels, strict=True) if a == b)
        equivalence_pct = matches / len(texts) * 100.0
        assert equivalence_pct >= 90.0, f"only {equivalence_pct:.1f}% equivalence on fixture"


class TestMaybeApplyDynamicQuantization:
    """maybe_apply_dynamic_quantization never corrupts model.onnx, whatever it decides."""

    @pytest.fixture
    def onnx_path(self, models_dir_with_sklearn_artifact: Path) -> Path:
        out_path = models_dir_with_sklearn_artifact / ONNX_MODEL_FILENAME
        export_to_onnx(models_dir_with_sklearn_artifact / MODEL_FILENAME, out_path)
        return out_path

    def test_raises_value_error_with_too_few_samples(self, onnx_path: Path) -> None:
        with pytest.raises(ValueError, match="at least 20"):
            maybe_apply_dynamic_quantization(onnx_path, ["short text"] * 5)

    def test_returns_a_well_formed_outcome_and_leaves_a_valid_onnx_file(
        self, onnx_path: Path
    ) -> None:
        texts = load_dataset(FIXTURE_PATH)["text"].tolist()
        assert len(texts) >= 20

        outcome = maybe_apply_dynamic_quantization(onnx_path, texts, warmup=1, runs=10)

        assert isinstance(outcome, QuantizationOutcome)
        assert isinstance(outcome.applied, bool)
        assert outcome.reason  # never empty -- always explains the decision
        assert outcome.baseline_p50_ms >= 0.0
        assert outcome.quantized_p50_ms >= 0.0
        assert 0.0 <= outcome.equivalence_pct <= 100.0

        # Whatever the decision, model.onnx must still be a loadable graph
        # afterward, and no leftover temp file.
        assert onnx_path.exists()
        ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        assert not onnx_path.with_suffix(".int8.tmp.onnx").exists()

    def test_applied_true_means_meta_records_it(
        self, onnx_path: Path, models_dir_with_sklearn_artifact: Path
    ) -> None:
        texts = load_dataset(FIXTURE_PATH)["text"].tolist()

        outcome = maybe_apply_dynamic_quantization(onnx_path, texts, warmup=1, runs=10)
        onnx_export._update_meta_after_quantization(onnx_path, outcome)

        meta = json.loads((models_dir_with_sklearn_artifact / ONNX_META_FILENAME).read_text())
        assert meta["quantized"] == outcome.applied
        assert meta["quantization_check"]["applied"] == outcome.applied
        assert meta["quantization_check"]["reason"] == outcome.reason


class TestCli:
    """python -m triagem.optimization.onnx_export."""

    def test_main_exports_and_runs_quantize_flag_end_to_end(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fitted_pipeline: Pipeline,
    ) -> None:
        models_dir = tmp_path / "cli_models"
        models_dir.mkdir()
        joblib.dump(fitted_pipeline, models_dir / MODEL_FILENAME)

        fake_settings = Settings(models_dir=models_dir, data_path=FIXTURE_PATH)
        monkeypatch.setattr(onnx_export, "get_settings", lambda: fake_settings)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "onnx_export",
                "--quantize",
                "--quantize-sample-n",
                "20",
                "--quantize-warmup",
                "1",
            ],
        )

        onnx_export.main()

        onnx_path = models_dir / ONNX_MODEL_FILENAME
        assert onnx_path.exists()
        meta = json.loads((models_dir / ONNX_META_FILENAME).read_text())
        assert "quantization_check" in meta

    def test_build_arg_parser_defaults(self) -> None:
        args = onnx_export._build_arg_parser().parse_args([])

        assert args.model_path is None
        assert args.out_path is None
        assert args.opset is None
        assert args.quantize is False
        assert args.quantize_sample_n == 100
        assert args.quantize_warmup == 20
        assert args.seed == 42
