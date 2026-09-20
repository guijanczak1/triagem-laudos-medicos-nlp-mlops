"""Tests for triagem.optimization.benchmark.measure_equivalence (T21).

Dedicated coverage for the sklearn-vs-onnx prediction-equivalence check --
the gate the task's acceptance criteria hold to a hard floor
(``state/backlog.json``: ``thresholds.prediction_equivalence_min_pct``,
99%). ``tests/optimization/test_benchmark.py`` covers the rest of
``benchmark()`` (latency measurement, CLI, report rendering); this file
isolates :func:`~triagem.optimization.benchmark.measure_equivalence` and
:class:`~triagem.optimization.benchmark.EquivalenceResult` so the
agreement math itself is verified independently of latency timing.

Real predictions come from the real T6 pipeline (fit on the small,
deterministic ``tests/fixtures/laudos_sample.csv`` fixture) exported to a
real ``model.onnx`` via T20's ``export_to_onnx`` -- never mocked scores --
the same pattern ``tests/optimization/test_onnx_export.py`` already
establishes for cross-backend agreement checks.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import joblib
import pytest
from sklearn.pipeline import Pipeline

from triagem.data.loaders import load_dataset
from triagem.models.pipeline import build_pipeline
from triagem.optimization.benchmark import (
    EQUIVALENCE_MIN_PCT,
    EquivalenceResult,
    measure_equivalence,
)
from triagem.optimization.onnx_export import export_to_onnx
from triagem.serving.predictor import Predictor
from triagem.training.train import MODEL_FILENAME

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "laudos_sample.csv"


@pytest.fixture(scope="module")
def fitted_pipeline() -> Pipeline:
    """Fit the real T6 pipeline on the small fixture once per test module."""
    df = load_dataset(FIXTURE_PATH)
    pipeline = build_pipeline(seed=42)
    pipeline.fit(df["text"].tolist(), df["label"].tolist())
    return pipeline


@pytest.fixture
def both_backends_models_dir(tmp_path: Path, fitted_pipeline: Pipeline) -> Path:
    """A models/ dir with a real model.joblib and a real, exported model.onnx."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(fitted_pipeline, models_dir / MODEL_FILENAME)
    export_to_onnx(models_dir / MODEL_FILENAME, models_dir / "model.onnx")
    return models_dir


@pytest.fixture(autouse=True)
def _clear_predictor_cache() -> Iterator[None]:
    """Every test gets an empty Predictor cache, and leaves none behind."""
    Predictor.clear_cache()
    yield
    Predictor.clear_cache()


class TestMeasureEquivalenceBothBackends:
    """measure_equivalence() with real sklearn + onnx predictors."""

    def test_compares_every_text_and_reports_real_counts(
        self, both_backends_models_dir: Path
    ) -> None:
        texts = load_dataset(FIXTURE_PATH)["text"].tolist()
        predictors = {
            "sklearn": Predictor.load(backend="sklearn", models_dir=both_backends_models_dir),
            "onnx": Predictor.load(backend="onnx", models_dir=both_backends_models_dir),
        }

        result = measure_equivalence(predictors, texts)

        assert isinstance(result, EquivalenceResult)
        assert result.n_compared == len(texts)
        assert 0 <= result.n_matches <= result.n_compared
        assert result.equivalence_pct == pytest.approx(result.n_matches / result.n_compared * 100.0)
        assert result.equivalence_min_pct == EQUIVALENCE_MIN_PCT

    def test_agreement_is_well_above_the_fixture_floor(
        self, both_backends_models_dir: Path
    ) -> None:
        """Same expectation test_onnx_export.py holds sklearn/onnx to on this
        fixture (>= 90%, looser than the 99% real-dataset floor -- a 54-row
        fixture can tip an occasional near-tied score across the
        float64/float32 boundary)."""
        texts = load_dataset(FIXTURE_PATH)["text"].tolist()
        predictors = {
            "sklearn": Predictor.load(backend="sklearn", models_dir=both_backends_models_dir),
            "onnx": Predictor.load(backend="onnx", models_dir=both_backends_models_dir),
        }

        result = measure_equivalence(predictors, texts)

        assert result.equivalence_pct >= 90.0, f"only {result.equivalence_pct:.1f}% equivalence"

    def test_passed_reflects_the_configured_floor(self, both_backends_models_dir: Path) -> None:
        texts = load_dataset(FIXTURE_PATH)["text"].tolist()
        predictors = {
            "sklearn": Predictor.load(backend="sklearn", models_dir=both_backends_models_dir),
            "onnx": Predictor.load(backend="onnx", models_dir=both_backends_models_dir),
        }

        lenient = measure_equivalence(predictors, texts, equivalence_min_pct=0.0)
        strict = measure_equivalence(predictors, texts, equivalence_min_pct=100.5)

        assert lenient.passed is True
        assert strict.passed is False


class TestMeasureEquivalenceEdgeCases:
    """measure_equivalence() stays well-defined outside the happy path."""

    def test_missing_onnx_predictor_returns_vacuous_pass(
        self, both_backends_models_dir: Path
    ) -> None:
        predictors = {
            "sklearn": Predictor.load(backend="sklearn", models_dir=both_backends_models_dir),
        }

        result = measure_equivalence(predictors, ["some text", "another text"])

        assert result.n_compared == 0
        assert result.n_matches == 0
        assert result.passed is True

    def test_missing_sklearn_predictor_returns_vacuous_pass(
        self, both_backends_models_dir: Path
    ) -> None:
        predictors = {
            "onnx": Predictor.load(backend="onnx", models_dir=both_backends_models_dir),
        }

        result = measure_equivalence(predictors, ["some text"])

        assert result.n_compared == 0
        assert result.passed is True

    def test_empty_texts_returns_vacuous_pass(self, both_backends_models_dir: Path) -> None:
        predictors = {
            "sklearn": Predictor.load(backend="sklearn", models_dir=both_backends_models_dir),
            "onnx": Predictor.load(backend="onnx", models_dir=both_backends_models_dir),
        }

        result = measure_equivalence(predictors, [])

        assert result.n_compared == 0
        assert result.passed is True

    def test_no_predictors_returns_vacuous_pass(self) -> None:
        result = measure_equivalence({}, ["text"])

        assert result.n_compared == 0
        assert result.passed is True
