"""TF-IDF + LogisticRegression baseline pipeline, exportable to ONNX (task T20).

``build_pipeline`` is the single place that defines the model architecture
for the classic (non-deep-learning) baseline. It deliberately sticks to
stock scikit-learn transformers/estimators with fixed constructor
hyperparameters -- no lambdas, no ``FunctionTransformer`` with a closure --
because task T20 converts this exact :class:`~sklearn.pipeline.Pipeline`
object to ONNX via ``skl2onnx``, which can only trace operators it has a
converter for. Any future change to this module must keep every step on
skl2onnx's supported-transformer list.

Revision note (T6/T7 review): the original ``RandomForestClassifier``
baseline was replaced with ``LogisticRegression`` after a real training run
on the full 14,438-sample dataset missed the quality gate by a wide margin
(macro_f1=0.4642 vs. required 0.80, recall_urgente=0.6303 vs. required
0.85) and re-tuning the forest (doubling ``n_estimators``, adjusting
``min_samples_leaf``) only closed a few points of that gap. The root cause
is a representation mismatch, not under-tuning: ``triagem.data.urgency``
labels urgency from an approximately *linear* combination of
length-normalized keyword counts plus a category prior, cut by quantile
thresholds. A ``RandomForestClassifier`` builds axis-aligned splits over
the ~20,000 sparse TF-IDF dimensions and has no efficient way to
reconstruct that linear ranking, while ``LogisticRegression`` fits a linear
decision boundary directly in that space -- a better-matched inductive bias
for this label. ``class_weight="balanced"`` is kept because urgency labels
are still imbalanced by construction (quantile cuts, not a 3-way even
split).

``LogisticRegression`` alone raised the real full-dataset numbers
(macro_f1=0.4642->0.5324, recall_urgente=0.6303->0.6944) but still missed
the gate. A grid search over the two fallback levers the task allowed --
``sublinear_tf`` and ``C`` (``max_features`` was also tried, up to 100k;
it never helped) -- found ``sublinear_tf=False`` with ``C=0.1`` as the best
real result on the full dataset: macro_f1=0.5716, recall_urgente=0.7280.
Every value of ``C`` above ~0.3 *hurt* both metrics (the model overfits the
~20,000-dim sparse space), and the metrics plateau hard past ``C~0.05-0.3``
and past ``max_features~20000`` -- more tuning along these two axes will
not close the remaining ~0.23 macro_f1 / ~0.12 recall_urgente gap. That gap
is consistent with the representation mismatch going deeper than
tree-vs-linear: ``compute_urgency_score`` (``triagem/data/urgency.py``) is
linear in *raw* keyword counts divided by word count, while TF-IDF is
L2-normalized (and, before this change, log-scaled) over the whole
20,000-dim vector -- a normalization the true label does not use, and that
neither RandomForest nor LogisticRegression can undo from the outside.
Closing the gap further is out of this task's scope (see T7's real
metrics.json and the harness report for the recommendation).
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

#: Step names, used by training/evaluation/export code to reach into the
#: fitted pipeline (e.g. ``pipeline[VECTORIZER_STEP]``) without hardcoding
#: positional indices.
VECTORIZER_STEP = "tfidf"
CLASSIFIER_STEP = "clf"


def build_pipeline(seed: int = 42, max_features: int = 20000) -> Pipeline:
    """Build the (unfitted) TF-IDF + LogisticRegression baseline pipeline.

    Args:
        seed: Random seed forwarded to ``LogisticRegression`` as
            ``random_state``. The default ``"lbfgs"`` solver is
            deterministic and ignores it, but it is always set so the
            pipeline stays reproducible if the solver is ever swapped for
            a stochastic one (e.g. ``"sag"``/``"saga"``). The same seed and
            training data always produce the same fitted model.
        max_features: Upper bound on the TF-IDF vocabulary size
            (``TfidfVectorizer(max_features=...)``).

    Returns:
        A scikit-learn ``Pipeline`` with two steps:
        ``VECTORIZER_STEP`` (``TfidfVectorizer``) followed by
        ``CLASSIFIER_STEP`` (``LogisticRegression``). Not yet fit --
        callers train it with ``pipeline.fit(texts, labels)``.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=False,
        strip_accents="unicode",
        lowercase=True,
        max_features=max_features,
    )
    classifier = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        C=0.1,
        random_state=seed,
        class_weight="balanced",
    )
    return Pipeline(steps=[(VECTORIZER_STEP, vectorizer), (CLASSIFIER_STEP, classifier)])
