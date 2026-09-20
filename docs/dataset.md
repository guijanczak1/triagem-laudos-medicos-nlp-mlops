# Dataset

This document describes the dataset the triage classifier is trained and
evaluated on: where it comes from, its license, its volume and exact
schema, how to rebuild it from scratch, and how to swap it for a real
clinical-report dataset in the future.

## Source

- **Corpus:** [`sebischair/Medical-Abstracts-TC-Corpus`](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)
- **Content:** English-language medical abstracts, each labeled with one of
  five clinical categories (`condition_label` / `condition_name`):
  1. neoplasms
  2. digestive system diseases
  3. nervous system diseases
  4. cardiovascular diseases
  5. general pathological conditions
- **Files consumed:** `medical_tc_train.csv`, `medical_tc_test.csv`,
  `medical_tc_labels.csv`, fetched from the corpus's `main` branch on
  GitHub's raw-content endpoint. No API key or authentication is required.

The corpus labels *disease category*, not *urgency*. This project's
`urgente`/`atencao`/`normal` label does not come from the corpus -- it is
derived by a heuristic documented in `docs/model_card.md`. Read that
document before trusting or reusing the label.

## License and attribution

**CC BY-SA 3.0** (Creative Commons Attribution-ShareAlike 3.0 Unported).

The raw files are redistributed unmodified in `data/raw/`, alongside a
generated `data/raw/LICENSE_DATASET.md` that records this same attribution
next to the data itself. Any derivative dataset built from this corpus
(including `data/processed/laudos.csv`) must keep this attribution and stay
under a compatible share-alike license if redistributed.

## Volume

- **14,438 total rows** combined from the original train (11,550) and test
  (2,888) splits into one processed file, each row carrying a `split` column
  that records its origin.
- Class balance in the processed label (`label`, derived not native) is
  targeted at quantiles `q=0.35`/`q=0.75` (see `docs/model_card.md`) but
  lands close to, not exactly on, those fractions because the underlying
  score is an integer keyword-count difference with many ties. Observed on
  the 20% stratified holdout (2,888 rows) used to compute
  `models/metrics.json`:

  | label     | count | share |
  |-----------|------:|------:|
  | `normal`   |   948 | 32.8% |
  | `atencao`  |  1132 | 39.2% |
  | `urgente`  |   808 | 28.0% |

  Exact per-run counts for the full dataset are written to
  `data/processed/urgency_thresholds.json`.

## Schema

`data/processed/laudos.csv` (the file `triagem.data.loaders.load_dataset`
reads):

| column            | type  | description |
|-------------------|-------|-------------|
| `id`              | str   | `"{split}-{index:06d}"`, e.g. `train-000123` |
| `text`            | str   | The abstract text (English, academic register) |
| `condition_label` | int   | Original corpus category id, 1-5 |
| `condition_name`  | str   | Human-readable name for `condition_label` |
| `urgency_score`   | float | Deterministic heuristic score (see model card) |
| `label`           | str   | One of `normal`, `atencao`, `urgente` -- the training target |
| `split`           | str   | `train` or `test`, the row's origin in the original corpus |

Only `text` and `label` are required by `load_dataset`/downstream training
code (`triagem.data.loaders.REQUIRED_COLUMNS`); the other columns are kept
for traceability and re-labeling experiments. `load_dataset` raises a
`DatasetSchemaError` with an actionable message if `text` is empty/missing
or `label` holds a value outside `{normal, atencao, urgente}`.

Downstream code never assumes the CSV lives on disk at a fixed path --
`Settings.data_path` is the single seam, and any file exposing at least
`text`/`label` columns loads with no code change (see "Swapping in a real
dataset" below).

## Reconstruction

The processed dataset is fully reproducible from the public corpus in two
steps:

```bash
python -m triagem.data.download          # writes data/raw/*.csv + LICENSE_DATASET.md
python -m triagem.data.build --seed 42    # writes data/processed/laudos.csv + urgency_thresholds.json
```

`download_raw` is idempotent (already-downloaded files are left untouched
unless `--force` is passed) and validates each file's header against the
expected schema before writing it, so a silently-changed upstream format
fails loudly instead of poisoning the dataset. `build_dataset` is a pure
function of the raw files and the (code-versioned) urgency lexicons: running
it twice on the same raw input produces byte-for-byte identical output.

## Swapping in a real dataset

`Settings.data_path` (env var `TRIAGEM_DATA_PATH`) is the only thing that
needs to change. To use a real clinical-report dataset instead of this
academic corpus:

1. Produce a CSV with at least a `text` column (report/abstract body) and a
   `label` column already in `{normal, atencao, urgente}` -- i.e. with
   *real*, not heuristic, urgency labels assigned by a qualified reviewer.
2. Point `TRIAGEM_DATA_PATH` (or `Settings.data_path`) at that file.
3. Re-run `python -m triagem.training.train --seed 42` -- no other code
   changes are required; `triagem.data.urgency` (the heuristic label
   generator) simply stops being invoked, since the new file already
   carries real labels.
4. Re-evaluate model performance and the quality gate
   (`triagem.training.evaluate.QualityGate`) against the new label
   distribution before trusting the numbers -- the current thresholds were
   calibrated against this heuristic dataset, not a clinical one.

Real clinical text also differs structurally from academic abstracts (see
"Limitations" in `docs/model_card.md`): shorter, more abbreviation-heavy,
often unstructured. Expect to revisit `TfidfVectorizer` hyperparameters
(`triagem.models.pipeline.build_pipeline`) once a real dataset is available.
