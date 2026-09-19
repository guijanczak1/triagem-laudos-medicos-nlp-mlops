"""Dataset loading contract consumed by every downstream ML/serving task.

``load_dataset`` and ``split_dataset`` are the single seam between the
processed CSV on disk (``triagem.data.build`` output, task T4) and every
consumer that trains, evaluates or serves on it (T6+). Swapping
``data/processed/laudos.csv`` for an external dataset with the same
``text``/``label`` columns must work with no change to any downstream code
-- only this module knows the file lives on disk at all.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from triagem.config import get_settings
from triagem.exceptions import DatasetSchemaError

logger = logging.getLogger(__name__)

#: Valid values for the ``label`` column (see ``triagem.data.urgency.UrgencyLabel``).
VALID_LABELS: tuple[str, ...] = ("normal", "atencao", "urgente")

#: Columns load_dataset requires. Intentionally a subset of
#: ``triagem.data.build.OUTPUT_COLUMNS``: only what downstream consumers
#: actually rely on is enforced here, so an external dataset with extra
#: columns (or missing the heuristic-specific ones) still loads fine.
REQUIRED_COLUMNS: tuple[str, ...] = ("text", "label")


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load and validate the processed triage dataset.

    Args:
        path: CSV path to load. Defaults to ``Settings.data_path``. Pointing
            this at a different CSV that exposes the same ``text``/``label``
            columns swaps the dataset with no code change anywhere else.

    Returns:
        The loaded DataFrame, unmodified beyond pandas' own CSV parsing.

    Raises:
        DatasetSchemaError: If the file does not exist, a required column is
            missing, ``text`` has an empty/missing value, or ``label``
            contains a value outside ``VALID_LABELS``.
    """
    resolved_path = path if path is not None else get_settings().data_path
    if not resolved_path.exists():
        raise DatasetSchemaError(
            f"{resolved_path} not found. Run `python -m triagem.data.build` first (task T4)."
        )

    df = pd.read_csv(resolved_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DatasetSchemaError(
            f"{resolved_path}: missing required column(s) {missing}. "
            f"Expected at least {list(REQUIRED_COLUMNS)}."
        )

    _validate_text_column(df, resolved_path)
    _validate_label_column(df, resolved_path)

    logger.info("loaded %d rows from %s", len(df), resolved_path)
    return df


def _validate_text_column(df: pd.DataFrame, path: Path) -> None:
    """Raise DatasetSchemaError if ``text`` has any empty or missing value."""
    text = df["text"]
    invalid = text.isna() | (text.astype(str).str.strip() == "")
    if bool(invalid.any()):
        bad_rows = df.index[invalid].tolist()[:5]
        raise DatasetSchemaError(
            f"{path}: column 'text' has {int(invalid.sum())} empty/missing value(s) "
            f"(e.g. row index {bad_rows}); every row must have non-empty text."
        )


def _validate_label_column(df: pd.DataFrame, path: Path) -> None:
    """Raise DatasetSchemaError if ``label`` has any value outside VALID_LABELS."""
    actual = set(df["label"].astype(str).unique())
    invalid_values = sorted(actual - set(VALID_LABELS))
    if invalid_values:
        raise DatasetSchemaError(
            f"{path}: column 'label' contains invalid value(s) {invalid_values}; "
            f"expected only {list(VALID_LABELS)}."
        )


def split_dataset(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into stratified train/test DataFrames by ``label``.

    Args:
        df: DataFrame with at least a ``label`` column (typically
            ``load_dataset()``'s output).
        test_size: Fraction of rows reserved for the test split.
        seed: Random seed forwarded to scikit-learn; the same ``df``,
            ``test_size`` and ``seed`` always produce the same split.

    Returns:
        ``(train_df, test_df)``, each with a fresh ``RangeIndex`` and the
        class proportions of ``label`` preserved (stratified split).

    Raises:
        DatasetSchemaError: If ``df`` has no ``label`` column.
    """
    if "label" not in df.columns:
        raise DatasetSchemaError("split_dataset: DataFrame has no 'label' column to stratify on.")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
