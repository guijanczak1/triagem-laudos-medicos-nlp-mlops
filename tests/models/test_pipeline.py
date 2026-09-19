"""Tests for triagem.models.pipeline.

Fits the baseline pipeline on the small, deterministic
``tests/fixtures/laudos_sample.csv`` fixture (54 real abstracts, all 3
labels present, no network) and checks: pipeline shape/hyperparameters,
fit/predict without error, determinism for a fixed seed, a sanity-level
holdout metric, and the ``predict_proba`` column-order contract that T20's
ONNX export relies on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from triagem.data.loaders import VALID_LABELS, load_dataset, split_dataset
from triagem.models.pipeline import CLASSIFIER_STEP, VECTORIZER_STEP, build_pipeline

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "laudos_sample.csv"


class TestBuildPipelineShape:
    """Structural/hyperparameter contract build_pipeline() must honor."""

    def test_returns_a_two_step_pipeline(self) -> None:
        pipeline = build_pipeline()

        assert isinstance(pipeline, Pipeline)
        assert [name for name, _ in pipeline.steps] == [VECTORIZER_STEP, CLASSIFIER_STEP]

    def test_vectorizer_hyperparameters(self) -> None:
        pipeline = build_pipeline(max_features=1234)
        vectorizer = pipeline.named_steps[VECTORIZER_STEP]

        assert isinstance(vectorizer, TfidfVectorizer)
        assert vectorizer.ngram_range == (1, 2)
        assert vectorizer.min_df == 2
        assert vectorizer.sublinear_tf is True
        assert vectorizer.strip_accents == "unicode"
        assert vectorizer.lowercase is True
        assert vectorizer.max_features == 1234

    def test_classifier_hyperparameters(self) -> None:
        pipeline = build_pipeline(seed=7)
        classifier = pipeline.named_steps[CLASSIFIER_STEP]

        assert isinstance(classifier, RandomForestClassifier)
        assert classifier.n_estimators == 300
        assert classifier.random_state == 7
        assert classifier.n_jobs == -1
        assert classifier.class_weight == "balanced"

    def test_no_lambdas_or_function_transformers(self) -> None:
        """Hard T20 prerequisite: only stock, skl2onnx-supported step types."""
        pipeline = build_pipeline()

        for _, step in pipeline.steps:
            assert step.__class__.__module__.startswith("sklearn."), (
                f"step {step!r} is not a stock scikit-learn transformer/estimator"
            )


class TestFitPredictOnFixture:
    """fit/predict on the small real-data fixture, deterministic and sane."""

    def _train_test_texts_labels(self) -> tuple[list[str], list[str], list[str], list[str]]:
        df = load_dataset(FIXTURE_PATH)
        train_df, test_df = split_dataset(df, test_size=0.2, seed=42)
        return (
            train_df["text"].tolist(),
            train_df["label"].tolist(),
            test_df["text"].tolist(),
            test_df["label"].tolist(),
        )

    def test_fit_predict_without_error(self) -> None:
        x_train, y_train, x_test, _y_test = self._train_test_texts_labels()
        pipeline = build_pipeline(seed=42)

        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        assert len(predictions) == len(x_test)
        assert set(predictions).issubset(set(VALID_LABELS))

    def test_predict_is_deterministic_for_the_same_seed(self) -> None:
        x_train, y_train, x_test, _y_test = self._train_test_texts_labels()

        pipeline_a = build_pipeline(seed=42)
        pipeline_a.fit(x_train, y_train)
        predictions_a = pipeline_a.predict(x_test)

        pipeline_b = build_pipeline(seed=42)
        pipeline_b.fit(x_train, y_train)
        predictions_b = pipeline_b.predict(x_test)

        assert list(predictions_a) == list(predictions_b)

    def test_sanity_metric_beats_trivial_floor(self) -> None:
        """Not a quality gate (that's T7) -- just proves the pipeline learns
        something on real text rather than predicting noise."""
        x_train, y_train, x_test, y_test = self._train_test_texts_labels()
        pipeline = build_pipeline(seed=42)

        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        accuracy = accuracy_score(y_test, predictions)
        assert accuracy >= 0.25

    def test_predict_proba_has_three_columns_in_classes_order(self) -> None:
        x_train, y_train, x_test, _y_test = self._train_test_texts_labels()
        pipeline = build_pipeline(seed=42)
        pipeline.fit(x_train, y_train)

        proba = pipeline.predict_proba(x_test)
        classifier = pipeline.named_steps[CLASSIFIER_STEP]

        assert proba.shape == (len(x_test), 3)
        assert list(classifier.classes_) == sorted(VALID_LABELS)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
