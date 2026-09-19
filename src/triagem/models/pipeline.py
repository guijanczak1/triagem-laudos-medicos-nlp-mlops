"""TF-IDF + RandomForest baseline pipeline, exportable to ONNX (task T20).

``build_pipeline`` is the single place that defines the model architecture
for the classic (non-deep-learning) baseline. It deliberately sticks to
stock scikit-learn transformers/estimators with fixed constructor
hyperparameters -- no lambdas, no ``FunctionTransformer`` with a closure --
because task T20 converts this exact :class:`~sklearn.pipeline.Pipeline`
object to ONNX via ``skl2onnx``, which can only trace operators it has a
converter for. Any future change to this module must keep every step on
skl2onnx's supported-transformer list.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

#: Step names, used by training/evaluation/export code to reach into the
#: fitted pipeline (e.g. ``pipeline[VECTORIZER_STEP]``) without hardcoding
#: positional indices.
VECTORIZER_STEP = "tfidf"
CLASSIFIER_STEP = "clf"


def build_pipeline(seed: int = 42, max_features: int = 20000) -> Pipeline:
    """Build the (unfitted) TF-IDF + RandomForest baseline pipeline.

    Args:
        seed: Random seed forwarded to ``RandomForestClassifier`` as
            ``random_state``. The same seed and training data always
            produce the same fitted model.
        max_features: Upper bound on the TF-IDF vocabulary size
            (``TfidfVectorizer(max_features=...)``).

    Returns:
        A scikit-learn ``Pipeline`` with two steps:
        ``VECTORIZER_STEP`` (``TfidfVectorizer``) followed by
        ``CLASSIFIER_STEP`` (``RandomForestClassifier``). Not yet fit --
        callers train it with ``pipeline.fit(texts, labels)``.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        max_features=max_features,
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    return Pipeline(steps=[(VECTORIZER_STEP, vectorizer), (CLASSIFIER_STEP, classifier)])
