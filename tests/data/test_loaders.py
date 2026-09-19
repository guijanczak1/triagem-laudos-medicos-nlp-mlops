"""Tests for triagem.data.loaders.

Uses the small, deterministic ``tests/fixtures/laudos_sample.csv`` fixture
(54 rows sampled from the real Medical Abstracts TC Corpus output, all 3
labels present) for the "real file" checks, plus tiny synthetic CSVs/
DataFrames written to ``tmp_path`` for schema-violation and error-path
checks. No network anywhere.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from triagem.config import Settings
from triagem.data.loaders import VALID_LABELS, load_dataset, split_dataset
from triagem.exceptions import DatasetSchemaError

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "laudos_sample.csv"


class TestLoadDatasetFixture:
    """load_dataset() against the small checked-in fixture."""

    def test_loads_fixture_with_expected_columns(self) -> None:
        df = load_dataset(FIXTURE_PATH)

        assert {"text", "label"}.issubset(df.columns)
        assert len(df) > 0

    def test_fixture_file_is_at_most_60_lines(self) -> None:
        lines = FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 60

    def test_fixture_contains_all_three_labels(self) -> None:
        df = load_dataset(FIXTURE_PATH)

        assert set(df["label"]) == set(VALID_LABELS)


class TestLoadDatasetPathResolution:
    """Default-path resolution via Settings.data_path."""

    def test_default_path_comes_from_settings_data_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = Settings(data_path=FIXTURE_PATH)
        monkeypatch.setattr("triagem.data.loaders.get_settings", lambda: settings)

        df = load_dataset()

        assert len(df) > 0

    def test_explicit_path_overrides_settings(self, tmp_path: Path) -> None:
        explicit = tmp_path / "explicit.csv"
        pd.DataFrame({"text": ["a report"], "label": ["normal"]}).to_csv(explicit, index=False)
        settings = Settings(data_path=FIXTURE_PATH)
        # get_settings is only consulted when path is None; wiring it here
        # proves an explicit path always wins, even with a different default.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("triagem.data.loaders.get_settings", lambda: settings)
            df = load_dataset(explicit)

        assert len(df) == 1
        assert df.loc[0, "text"] == "a report"

    def test_swapping_csv_for_external_dataset_needs_no_code_change(self, tmp_path: Path) -> None:
        """An external CSV exposing only text/label (plus unrelated columns) still loads."""
        external = tmp_path / "external.csv"
        pd.DataFrame(
            {
                "text": ["Some clinical note.", "Another one here."],
                "label": ["normal", "urgente"],
                "source_system": ["external_ehr", "external_ehr"],
            }
        ).to_csv(external, index=False)

        df = load_dataset(external)

        assert list(df["label"]) == ["normal", "urgente"]
        assert "source_system" in df.columns


class TestLoadDatasetErrors:
    """Schema violations raise DatasetSchemaError with an actionable message."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.csv"

        with pytest.raises(DatasetSchemaError, match="not found"):
            load_dataset(missing)

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        pd.DataFrame({"text": ["ok"]}).to_csv(path, index=False)

        with pytest.raises(DatasetSchemaError, match="missing required column"):
            load_dataset(path)

    def test_empty_text_value_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        pd.DataFrame({"text": ["ok", ""], "label": ["normal", "urgente"]}).to_csv(path, index=False)

        with pytest.raises(DatasetSchemaError, match="empty/missing value"):
            load_dataset(path)

    def test_missing_text_value_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        pd.DataFrame({"text": ["ok", None], "label": ["normal", "urgente"]}).to_csv(
            path, index=False
        )

        with pytest.raises(DatasetSchemaError, match="empty/missing value"):
            load_dataset(path)

    def test_invalid_label_value_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        pd.DataFrame({"text": ["ok", "ok too"], "label": ["normal", "unknown"]}).to_csv(
            path, index=False
        )

        with pytest.raises(DatasetSchemaError, match="invalid value"):
            load_dataset(path)


class TestSplitDataset:
    """split_dataset(): stratified, deterministic, index-reset."""

    def test_split_sizes_sum_to_original(self) -> None:
        df = load_dataset(FIXTURE_PATH)

        train_df, test_df = split_dataset(df, test_size=0.2, seed=42)

        assert len(train_df) + len(test_df) == len(df)

    def test_split_is_stratified_by_label(self) -> None:
        df = load_dataset(FIXTURE_PATH)

        train_df, test_df = split_dataset(df, test_size=0.2, seed=42)

        full_props = df["label"].value_counts(normalize=True)
        train_props = train_df["label"].value_counts(normalize=True)
        test_props = test_df["label"].value_counts(normalize=True)
        for label in VALID_LABELS:
            assert train_props[label] == pytest.approx(full_props[label], abs=0.15)
            assert test_props[label] == pytest.approx(full_props[label], abs=0.15)

    def test_split_is_deterministic_for_the_same_seed(self) -> None:
        df = load_dataset(FIXTURE_PATH)

        train_1, test_1 = split_dataset(df, seed=42)
        train_2, test_2 = split_dataset(df, seed=42)

        pd.testing.assert_frame_equal(train_1, train_2)
        pd.testing.assert_frame_equal(test_1, test_2)

    def test_different_seeds_can_produce_different_splits(self) -> None:
        df = load_dataset(FIXTURE_PATH)

        train_a, _ = split_dataset(df, seed=1)
        train_b, _ = split_dataset(df, seed=2)

        assert train_a["text"].tolist() != train_b["text"].tolist()

    def test_test_size_controls_split_proportions(self) -> None:
        df = load_dataset(FIXTURE_PATH)

        train_df, test_df = split_dataset(df, test_size=0.3, seed=42)

        assert len(test_df) == pytest.approx(len(df) * 0.3, abs=1)

    def test_indices_are_reset_on_both_splits(self) -> None:
        df = load_dataset(FIXTURE_PATH)

        train_df, test_df = split_dataset(df, seed=42)

        assert list(train_df.index) == list(range(len(train_df)))
        assert list(test_df.index) == list(range(len(test_df)))

    def test_missing_label_column_raises(self) -> None:
        df = pd.DataFrame({"text": ["a", "b", "c"]})

        with pytest.raises(DatasetSchemaError, match="no 'label' column"):
            split_dataset(df)
