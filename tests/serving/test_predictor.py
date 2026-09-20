"""Tests for triagem.serving.predictor.

Trains a real (small) pipeline in memory on the deterministic
``tests/fixtures/laudos_sample.csv`` fixture and persists it as a genuine
``model.joblib`` under ``tmp_path`` -- no artifact on the real ``models/``
directory is required or touched. Covers: the backend-agnostic
``Prediction``/``metadata`` contract, the lazy-singleton load cache, and the
``onnx`` backend's ``ModelArtifactNotFound`` guard (the artifact only
exists after task T20 runs).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import joblib
import pytest

from triagem.data.loaders import VALID_LABELS, load_dataset
from triagem.exceptions import ModelArtifactNotFound
from triagem.models.pipeline import build_pipeline
from triagem.serving.predictor import Prediction, Predictor
from triagem.training.train import LABEL_ENCODER_FILENAME, METRICS_FILENAME, MODEL_FILENAME

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "laudos_sample.csv"
TRAINED_AT = "2024-01-01T00:00:00+00:00"


@pytest.fixture(autouse=True)
def _clear_predictor_cache() -> Iterator[None]:
    """Every test gets an empty Predictor cache, and leaves none behind."""
    Predictor.clear_cache()
    yield
    Predictor.clear_cache()


@pytest.fixture(scope="module")
def _fitted_pipeline() -> object:
    """Fit the real T6 pipeline on the small fixture once per test module."""
    df = load_dataset(FIXTURE_PATH)
    pipeline = build_pipeline(seed=42)
    pipeline.fit(df["text"].tolist(), df["label"].tolist())
    return pipeline


@pytest.fixture
def sklearn_models_dir(tmp_path: Path, _fitted_pipeline: object) -> Path:
    """A models/ dir with a real, in-memory-trained sklearn artifact + metadata."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(_fitted_pipeline, models_dir / MODEL_FILENAME)
    (models_dir / METRICS_FILENAME).write_text(
        json.dumps({"trained_at": TRAINED_AT, "macro_f1": 0.6}), encoding="utf-8"
    )
    (models_dir / LABEL_ENCODER_FILENAME).write_text(
        json.dumps({"classes": sorted(VALID_LABELS)}), encoding="utf-8"
    )
    return models_dir


@pytest.fixture
def predictor(sklearn_models_dir: Path) -> Predictor:
    return Predictor.load(backend="sklearn", models_dir=sklearn_models_dir)


class TestLoadSklearnBackend:
    """Predictor.load() for the 'sklearn' backend."""

    def test_load_returns_predictor_with_sklearn_backend(self, sklearn_models_dir: Path) -> None:
        result = Predictor.load(backend="sklearn", models_dir=sklearn_models_dir)

        assert isinstance(result, Predictor)
        assert result.metadata.backend == "sklearn"

    def test_load_raises_model_artifact_not_found_when_model_joblib_missing(
        self, tmp_path: Path
    ) -> None:
        empty_dir = tmp_path / "empty_models"
        empty_dir.mkdir()

        with pytest.raises(ModelArtifactNotFound) as exc_info:
            Predictor.load(backend="sklearn", models_dir=empty_dir)

        message = str(exc_info.value)
        assert "model.joblib" in message
        assert "triagem.training.train" in message

    def test_load_is_a_lazy_singleton_no_repeated_disk_io(
        self, sklearn_models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original_load = joblib.load
        call_count = {"n": 0}

        def counting_load(path: Path, *args: object, **kwargs: object) -> object:
            call_count["n"] += 1
            return original_load(path, *args, **kwargs)

        monkeypatch.setattr("triagem.serving.predictor.joblib.load", counting_load)

        first = Predictor.load(backend="sklearn", models_dir=sklearn_models_dir)
        second = Predictor.load(backend="sklearn", models_dir=sklearn_models_dir)

        assert first is second
        assert call_count["n"] == 1

    def test_load_resolves_backend_from_settings_when_omitted(
        self, sklearn_models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from triagem.config import Settings

        monkeypatch.setattr(
            "triagem.serving.predictor.get_settings",
            lambda: Settings(model_backend="sklearn", models_dir=sklearn_models_dir),
        )

        result = Predictor.load()

        assert result.metadata.backend == "sklearn"


class TestPredictContract:
    """predict() / predict_batch() must honor the Prediction contract."""

    def test_predict_returns_prediction_with_expected_contract(self, predictor: Predictor) -> None:
        result = predictor.predict(
            "Acute severe emergency with critical organ failure and hemorrhage."
        )

        assert isinstance(result, Prediction)
        assert result.backend == "sklearn"
        assert set(result.scores) == set(VALID_LABELS)
        assert result.label in VALID_LABELS
        assert sum(result.scores.values()) == pytest.approx(1.0, abs=1e-6)
        assert result.latency_ms >= 0.0

    def test_predict_label_is_the_argmax_of_scores(self, predictor: Predictor) -> None:
        result = predictor.predict("Chronic mild stable follow-up, routine screening.")

        best_label = max(result.scores, key=lambda label: result.scores[label])
        assert result.label == best_label

    def test_predict_batch_returns_one_prediction_per_text_in_order(
        self, predictor: Predictor
    ) -> None:
        texts = [
            "Acute severe emergency with critical organ failure.",
            "Chronic mild stable follow-up, routine screening.",
            "Malignant metastatic carcinoma with sepsis and shock.",
        ]

        results = predictor.predict_batch(texts)

        assert len(results) == len(texts)
        for result in results:
            assert isinstance(result, Prediction)
            assert result.backend == "sklearn"
            assert set(result.scores) == set(VALID_LABELS)
            assert sum(result.scores.values()) == pytest.approx(1.0, abs=1e-6)

    def test_predict_batch_empty_input_returns_empty_list(self, predictor: Predictor) -> None:
        assert predictor.predict_batch([]) == []


class TestMetadata:
    """Predictor.metadata must expose backend, model_version, trained_at, classes."""

    def test_metadata_exposes_all_required_fields(self, predictor: Predictor) -> None:
        metadata = predictor.metadata

        assert metadata.backend == "sklearn"
        assert metadata.trained_at == TRAINED_AT
        assert metadata.model_version == TRAINED_AT
        assert set(metadata.classes) == set(VALID_LABELS)

    def test_model_version_falls_back_to_unknown_without_metrics_json(
        self, tmp_path: Path, _fitted_pipeline: object
    ) -> None:
        models_dir = tmp_path / "models_no_metrics"
        models_dir.mkdir()
        joblib.dump(_fitted_pipeline, models_dir / MODEL_FILENAME)

        result = Predictor.load(backend="sklearn", models_dir=models_dir)

        assert result.metadata.trained_at is None
        assert result.metadata.model_version == "unknown"


class TestOnnxBackendWithoutArtifact:
    """The 'onnx' backend must fail loudly (never fall back) before T20 runs."""

    def test_load_onnx_without_artifact_raises_with_clear_instruction(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        with pytest.raises(ModelArtifactNotFound) as exc_info:
            Predictor.load(backend="onnx", models_dir=models_dir)

        message = str(exc_info.value)
        assert "model.onnx" in message
        assert "onnx_export" in message
        assert "T20" in message
        assert "sklearn" in message  # points to the working alternative

    def test_onnx_missing_artifact_never_falls_back_to_sklearn(
        self, sklearn_models_dir: Path
    ) -> None:
        """A valid model.joblib sitting right next to it changes nothing."""
        with pytest.raises(ModelArtifactNotFound):
            Predictor.load(backend="onnx", models_dir=sklearn_models_dir)


class TestUnknownBackend:
    def test_load_rejects_an_unsupported_backend_value(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown model backend"):
            Predictor.load(backend="tensorflow", models_dir=tmp_path)  # type: ignore[arg-type]
