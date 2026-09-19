"""Build the processed triage dataset from the raw Medical Abstracts TC Corpus.

Reads the raw CSVs downloaded by ``triagem.data.download`` (task T3),
applies the urgency heuristic in ``triagem.data.urgency`` (keyword lexicon +
clinical category as a secondary tie-breaker -- decided WITH the user, see
``state/mlet-tech-challenge-fase-3/answers.md`` A1), and writes the
combined, labeled dataset to ``data/processed/laudos.csv``.

The urgency label produced here is a DIDACTIC heuristic, not a validated
clinical classification -- see ``docs/model_card.md`` (task T8).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from triagem.config import Settings, get_settings
from triagem.data.urgency import (
    THRESHOLDS_FILENAME,
    compute_urgency_score,
    fit_thresholds,
    map_urgency,
    save_thresholds,
)
from triagem.exceptions import DatasetSchemaError

logger = logging.getLogger(__name__)

_TRAIN_FILENAME = "medical_tc_train.csv"
_TEST_FILENAME = "medical_tc_test.csv"
_LABELS_FILENAME = "medical_tc_labels.csv"

#: Exact output schema required by T4's acceptance criteria.
OUTPUT_COLUMNS: tuple[str, ...] = (
    "id",
    "text",
    "condition_label",
    "condition_name",
    "urgency_score",
    "label",
    "split",
)


def _require_file(path: Path) -> None:
    """Raise DatasetSchemaError with an actionable message if ``path`` is missing."""
    if not path.exists():
        raise DatasetSchemaError(
            f"{path} not found. Run `python -m triagem.data.download` first (task T3)."
        )


def _load_labels(raw_dir: Path) -> pd.DataFrame:
    """Load and validate ``medical_tc_labels.csv`` (condition_label -> condition_name)."""
    path = raw_dir / _LABELS_FILENAME
    _require_file(path)
    labels = pd.read_csv(path)
    missing = {"condition_label", "condition_name"} - set(labels.columns)
    if missing:
        raise DatasetSchemaError(f"{_LABELS_FILENAME}: missing column(s) {sorted(missing)}.")
    return labels


def _load_split(raw_dir: Path, filename: str, split: str, labels: pd.DataFrame) -> pd.DataFrame:
    """Load one raw split (train/test), join condition names, score and id rows.

    Args:
        raw_dir: Directory with the raw corpus CSVs.
        filename: Raw CSV filename (``medical_tc_train.csv`` or
            ``medical_tc_test.csv``).
        split: Split label written to the output ``split`` column
            (``"train"`` or ``"test"``).
        labels: The ``condition_label -> condition_name`` lookup table.

    Returns:
        A DataFrame with columns
        ``id, text, condition_label, condition_name, urgency_score, split``.

    Raises:
        DatasetSchemaError: If the raw file is missing, has an unexpected
            schema, or references a ``condition_label`` absent from
            ``labels``.
    """
    path = raw_dir / filename
    _require_file(path)
    raw = pd.read_csv(path)
    missing = {"condition_label", "medical_abstract"} - set(raw.columns)
    if missing:
        raise DatasetSchemaError(f"{filename}: missing column(s) {sorted(missing)}.")

    merged = raw.merge(labels, on="condition_label", how="left")
    unmatched = merged["condition_name"].isna()
    if bool(unmatched.any()):
        unknown = sorted(merged.loc[unmatched, "condition_label"].unique().tolist())
        raise DatasetSchemaError(
            f"{filename}: condition_label value(s) {unknown} are absent from {_LABELS_FILENAME}."
        )

    merged = merged.rename(columns={"medical_abstract": "text"})
    merged["split"] = split
    merged["id"] = [f"{split}-{i:06d}" for i in range(len(merged))]
    merged["urgency_score"] = [
        compute_urgency_score(text, int(condition_label))
        for text, condition_label in zip(merged["text"], merged["condition_label"], strict=True)
    ]
    return merged[["id", "text", "condition_label", "condition_name", "urgency_score", "split"]]


def build_dataset(raw_dir: Path, out_path: Path, seed: int = 42) -> Path:
    """Build the processed, urgency-labeled dataset and write it to ``out_path``.

    Loads the raw train/test corpus and label names from ``raw_dir``, scores
    every abstract with ``compute_urgency_score`` (keyword lexicon +
    ``CONDITION_PRIOR`` as a secondary tie-breaker), fits the fixed-quantile
    thresholds (q=0.35, q=0.75) on the TRAIN split's scores only, and reuses
    those same thresholds to label the TEST split -- no leakage. Thresholds
    are persisted next to ``out_path`` as ``urgency_thresholds.json``.

    Args:
        raw_dir: Directory holding the raw corpus CSVs
            (``triagem.data.download`` output).
        out_path: Destination CSV path. Columns: id, text, condition_label,
            condition_name, urgency_score, label, split.
        seed: Recorded alongside the fitted thresholds for provenance; the
            heuristic itself is deterministic and consumes no randomness, so
            the same seed (or any seed) always reproduces the same CSV.

    Returns:
        ``out_path``.

    Raises:
        DatasetSchemaError: If a raw file is missing or has an unexpected
            schema.
    """
    labels = _load_labels(raw_dir)
    train_df = _load_split(raw_dir, _TRAIN_FILENAME, "train", labels)
    test_df = _load_split(raw_dir, _TEST_FILENAME, "test", labels)

    thresholds = fit_thresholds(train_df["urgency_score"].tolist())

    train_df = train_df.assign(
        label=[map_urgency(score, thresholds) for score in train_df["urgency_score"]]
    )
    test_df = test_df.assign(
        label=[map_urgency(score, thresholds) for score in test_df["urgency_score"]]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_thresholds(thresholds, out_path.parent / THRESHOLDS_FILENAME, seed=seed)

    combined = pd.concat([train_df, test_df], ignore_index=True)[list(OUTPUT_COLUMNS)]
    combined.to_csv(out_path, index=False)
    logger.info(
        "wrote %d rows (%d train, %d test) to %s",
        len(combined),
        len(train_df),
        len(test_df),
        out_path,
    )
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``python -m triagem.data.build``."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the processed, urgency-labeled triage dataset from the raw "
            "Medical Abstracts TC Corpus (heuristic label, see docs/model_card.md)."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory with the raw corpus CSVs (defaults to Settings.raw_dir).",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Destination CSV path (defaults to Settings.data_path).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed recorded alongside the fitted thresholds (default: 42).",
    )
    return parser


def main() -> None:
    """CLI entry point: ``python -m triagem.data.build --seed 42``."""
    from triagem.logging_conf import setup_logging

    setup_logging()
    args = _build_arg_parser().parse_args()
    settings: Settings = get_settings()
    raw_dir = args.raw_dir if args.raw_dir is not None else settings.raw_dir
    out_path = args.out_path if args.out_path is not None else settings.data_path
    written = build_dataset(raw_dir=raw_dir, out_path=out_path, seed=args.seed)
    print(f"dataset: {written}")


if __name__ == "__main__":
    main()
