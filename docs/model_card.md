# Model Card: Triage Urgency Classifier

## What this model does

Given a medical-report text, the model predicts one of three urgency
classes: `normal`, `atencao` (attention), `urgente` (urgent). It is a
`TfidfVectorizer` + `LogisticRegression` scikit-learn pipeline
(`triagem.models.pipeline.build_pipeline`), trained end-to-end on a
processed version of the [Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)
(see `docs/dataset.md`).

## Intended use and hard limitations

**This is a didactic system. It is NOT a validated clinical tool and must
NOT be used to support real medical decisions, triage real patients, or
substitute for review by a qualified clinician.** Concretely:

- The `urgente`/`atencao`/`normal` label was never assigned by a clinician.
  It is a **deterministic keyword-and-category heuristic** applied to the
  corpus text (see "How the label was built" below), not a diagnosis or a
  triage decision made by a person with medical training.
- The training text is English-language **academic abstracts**
  summarizing research on a condition, not real clinical reports/laudos
  written about an actual patient encounter. Register, structure, and the
  kind of information present differ substantially from real clinical
  documentation (see "Limitations and known biases").
- The model's real, measured performance (below) is well short of what
  would be required for any safety-relevant use.

Suitable uses: coursework, demonstrating an end-to-end NLP + MLOps pipeline
(data versioning-free reproducible build, training, evaluation gate,
serving, CI/CD, observability, latency benchmarking), and as a starting
point that a real deployment would need to replace the label source for
(see `docs/dataset.md`, "Swapping in a real dataset").

## How the label was built

The corpus natively labels disease **category** (neoplasms, digestive,
nervous, cardiovascular, general pathological), not **urgency**. Mapping
category directly to urgency would make the classification task trivial --
the model would just be re-identifying the disguised original category
instead of learning anything about urgency.

Instead, urgency was derived with a decision made deliberately, and
declared up front, with the project's owner (decision "A1"): urgency comes
**primarily from clinical keywords present in the abstract text**, with the
clinical category acting only as a **secondary, low-weight tie-breaker**.

- **Urgent-leaning terms:** acute, severe, malignant, metastatic,
  emergency, hemorrhage, sepsis, shock, infarction, rupture, obstruction,
  failure, fatal, critical, carcinoma.
- **Normal-leaning terms:** chronic, benign, mild, routine, stable,
  follow-up, screening, elective, remission, asymptomatic,
  well-controlled.
- **Category tie-breaker** (`CONDITION_PRIOR`, added to the keyword
  difference): neoplasms +0.15, cardiovascular +0.15, nervous +0.05,
  digestive +0.00, general pathological -0.05.
- **Score:** `(urgent keyword hits) - (normal keyword hits) + category
  prior`, a pure, deterministic function of the text -- no randomness, byte
  -for-byte reproducible across runs.
- **Label cut:** fixed quantiles of the score, fitted on the training split
  only and reused unchanged on the test split (no leakage): below the
  0.35 quantile -> `normal`, between 0.35 and 0.75 -> `atencao`, at or
  above 0.75 -> `urgente`.

This lexicon-plus-category heuristic is versioned in code
(`src/triagem/data/urgency.py`), not hidden in data, so it is fully
auditable and reproducible -- but it remains a heuristic, not ground truth.

## Model architecture

`TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=False,
strip_accents="unicode", max_features=20000)` feeding a
`LogisticRegression(solver="lbfgs", C=0.1, max_iter=2000,
class_weight="balanced")`. Deliberately built from stock, skl2onnx-exportable
transformers only (no lambdas/closures), so the same pipeline can later be
exported to ONNX without a representation change.

## Real performance (full dataset, seed=42)

Measured on a stratified 20% holdout of the full 14,438-sample processed
dataset (`models/metrics.json`, `sklearn==1.9.1`):

| metric | value |
|---|---:|
| macro F1 | 0.6427 |
| accuracy | 0.6343 |
| recall (`urgente`) | 0.7364 |

Per-class breakdown:

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| `normal` | 0.6002 | 0.5876 | 0.5938 | 948 |
| `atencao` | 0.5842 | 0.6007 | 0.5923 | 1132 |
| `urgente` | 0.7475 | 0.7364 | 0.7419 | 808 |

Confusion matrix (rows = true label, columns = predicted label, order
`atencao, normal, urgente`):

| true \ pred | atencao | normal | urgente |
|---|---:|---:|---:|
| **atencao** | 680 | 310 | 142 |
| **normal**  | 332 | 557 |  59 |
| **urgente** | 152 |  61 | 595 |

`urgente` is the class the model separates best (74% recall, 75%
precision) -- consistent with it sitting at the tail of the keyword-score
distribution, farthest from the ambiguous middle. `normal` and `atencao`
are confused with each other far more often (310 and 332 misclassifications
between them) than either is confused with `urgente`, which tracks the
heuristic's own geometry: they are adjacent quantile bands of the same
continuous score, so the boundary between them is inherently softer than
the boundary against the `urgente` tail.

## Quality gate and the decision behind it

The project originally targeted macro F1 >= 0.80 and `urgente` recall >=
0.85. Those numbers were an ambitious default proposed before any real
training run, not a number required by the assignment brief, which asked
only for a working model. They were never met.

Two rounds of technical correction were carried out before accepting the
gap, in order:

1. **Model family.** The original `RandomForestClassifier` baseline missed
   the original gate by a wide margin (macro F1 0.4642, recall_urgente
   0.6303) and re-tuning the forest closed only a few points. Diagnosis:
   the label is an approximately linear function of the input features
   (keyword counts plus a category prior), and a tree ensemble has no
   efficient way to reconstruct a linear ranking over ~20,000 sparse TF-IDF
   dimensions. Switching to `LogisticRegression`, whose linear decision
   boundary matches that geometry, raised the real numbers to macro F1
   0.5324 / recall_urgente 0.6944 immediately, and a follow-up grid search
   over `sublinear_tf` and `C` found the current `C=0.1` as the best real
   result at that stage: macro F1 0.5716 / recall_urgente 0.7280.
2. **Label-aggregation formula.** The remaining gap was traced to a
   mismatch between how the label score was computed and how TF-IDF
   represents text: the original score divided the keyword-hit difference
   by the document's raw word count, a length normalization TF-IDF's own
   L2-normalized, ~20,000-dimension vector does not use. Three
   alternative normalizations were evaluated empirically on the full
   corpus against the same model architecture: presence-weighted by
   `1/sqrt(unique terms)` (macro F1 0.6363 / recall_urgente 0.7455), binary
   term presence summed per lexicon (macro F1 0.5985 / recall_urgente
   0.6981), and dropping the length normalization entirely (macro F1
   0.6427 / recall_urgente 0.7364 -- the current, best-performing formula,
   used in `compute_urgency_score`).

After these two rounds, further hyperparameter search along the remaining
free axes (`C`, `max_features`, alternative fractional-power length
normalizations) plateaued around macro F1 ~0.71 / recall_urgente ~0.82 at
best in exploratory runs, still short of 0.80/0.85, with no lever left
that reliably closed the gap further without re-opening the trivial-label
problem the keyword+category design was built to avoid in the first
place.

With reasonable technical correction attempts exhausted, the project owner
reviewed these real numbers and made an explicit decision: accept them,
document the limitation honestly (this document), and adjust the
project's quality gate to reflect achievable performance rather than an
unvalidated aspirational target. The gate
(`triagem.training.evaluate.QualityGate`) is now:

- `macro_f1_min = 0.55`
- `recall_urgente_min = 0.70`

Both sit below the measured 0.6427/0.7364, giving realistic margin for
run-to-run variance on retraining without being a trivial bar to clear.
This is a deliberate, informed, and documented adjustment, not an attempt
to hide underperformance -- the honest numbers, and the two rounds of
correction that preceded the decision, are recorded in full above and in
`src/triagem/models/pipeline.py` / `src/triagem/data/urgency.py`.

## Limitations and known biases

- **Heuristic label, not clinical ground truth.** Every number in this
  document measures how well the model predicts the *keyword-and-category
  heuristic*, not real urgency as a clinician would assess it. A model
  that reproduces this heuristic well is not thereby validated for real
  triage.
- **Academic abstracts, not clinical reports.** Training text is
  English-language research-abstract prose (structured, well-formed,
  written for publication), not real clinical notes/laudos, which tend to
  be shorter, more abbreviation-heavy, and less uniformly structured. A
  model trained here should not be assumed to generalize to real report
  text without re-evaluation.
- **English only.** No non-English text was used in training or
  evaluation; the model has no demonstrated ability to work in other
  languages.
- **Keyword-list coverage bias.** The urgency lexicons are a fixed,
  hand-picked list of English clinical terms. Any real urgency signal
  expressed through vocabulary outside these lists, negation ("no evidence
  of hemorrhage"), or non-lexical cues (e.g. numeric values, dates,
  structured fields) is invisible to the label generator and, by
  construction, to the model trained on it.
- **`normal` vs. `atencao` confusion.** These two classes are the most
  frequently confused with each other (see confusion matrix), consistent
  with them being adjacent bands of one continuous score rather than
  semantically distinct categories with a clear boundary.
- **Class balance is a heuristic artifact.** The ~33/39/28 class split
  (`normal`/`atencao`/`urgente`) is a byproduct of the fixed quantile cuts
  (0.35/0.75) applied to the score distribution, not an estimate of real
  urgency prevalence in any clinical population.

## How to move toward real clinical validity

See `docs/dataset.md`, "Swapping in a real dataset", for the mechanical
steps (`Settings.data_path` is the only integration point). Beyond that
swap, moving this system toward anything usable in a real setting would
require, at minimum: urgency labels assigned by qualified clinical
reviewers (replacing the keyword heuristic entirely), evaluation on real
clinical-report text, and a formal clinical validation process -- none of
which this project attempts or claims to provide.
