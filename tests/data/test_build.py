"""Tests for triagem.data.build.

Fixture-based tests (no network) cover schema, error handling, no-leakage
threshold reuse and byte-for-byte determinism. A single network-marked test
exercises the real pipeline end-to-end (download + build) and asserts the
PDF's minimum volume (>= 2000 rows) and the 15%-per-class floor on real
data; it is deselected by default (see ``-m "not network"`` in
pyproject.toml).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

from triagem.config import Settings
from triagem.data.build import OUTPUT_COLUMNS, build_dataset, main
from triagem.data.download import download_raw
from triagem.data.urgency import THRESHOLDS_FILENAME, fit_thresholds, load_thresholds
from triagem.exceptions import DatasetSchemaError

_LABELS_CSV = (
    "condition_label,condition_name\n"
    "1,neoplasms\n"
    "2,digestive system diseases\n"
    "3,nervous system diseases\n"
    "4,cardiovascular diseases\n"
    "5,general pathological conditions\n"
)

# 12 synthetic train rows and 6 synthetic test rows: a mix of clearly
# urgent, clearly normal and neutral abstracts across the 5 categories, big
# enough for quantile-based thresholds to produce all three labels.
_TRAIN_ROWS = [
    (1, "Acute malignant carcinoma with metastatic spread and organ failure."),
    (1, "Severe emergency hemorrhage requiring critical intervention."),
    (1, "Routine screening detected a small benign lesion, patient stable."),
    (2, "Chronic mild condition managed with routine follow-up visits."),
    (2, "Elective procedure for a well-controlled digestive condition."),
    (2, "Acute obstruction with shock, emergency surgery required."),
    (3, "Stable remission after treatment, asymptomatic on follow-up."),
    (3, "Critical rupture with severe hemorrhage and fatal outcome risk."),
    (4, "Chronic stable cardiovascular condition, benign and asymptomatic."),
    (4, "Acute myocardial infarction, severe and critical, emergency care."),
    (5, "Mild routine pathological finding, elective screening only."),
    (5, "Severe sepsis with critical organ failure and shock."),
]
_TEST_ROWS = [
    (1, "Metastatic carcinoma, severe and fatal prognosis."),
    (2, "Routine elective follow-up, chronic but well-controlled."),
    (3, "Asymptomatic remission, stable and benign."),
    (4, "Acute infarction with critical emergency shock."),
    (5, "Mild benign screening finding, routine follow-up."),
    (1, "Severe malignant rupture requiring emergency intervention."),
]


def _write_raw_corpus(raw_dir: Path) -> None:
    """Write a small, self-contained raw corpus (train/test/labels) to raw_dir."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "medical_tc_labels.csv").write_text(_LABELS_CSV, encoding="utf-8")

    train_df = pd.DataFrame(_TRAIN_ROWS, columns=["condition_label", "medical_abstract"])
    train_df.to_csv(raw_dir / "medical_tc_train.csv", index=False)

    test_df = pd.DataFrame(_TEST_ROWS, columns=["condition_label", "medical_abstract"])
    test_df.to_csv(raw_dir / "medical_tc_test.csv", index=False)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A small synthetic raw corpus, self-contained and network-free."""
    directory = tmp_path / "raw"
    _write_raw_corpus(directory)
    return directory


class TestBuildDatasetSchema:
    """Output schema/content of build_dataset."""

    def test_output_columns_match_exact_schema(self, raw_dir: Path, tmp_path: Path) -> None:
        out_path = tmp_path / "processed" / "laudos.csv"

        build_dataset(raw_dir, out_path)

        df = pd.read_csv(out_path)
        assert list(df.columns) == list(OUTPUT_COLUMNS)

    def test_row_count_equals_train_plus_test(self, raw_dir: Path, tmp_path: Path) -> None:
        out_path = tmp_path / "laudos.csv"

        build_dataset(raw_dir, out_path)

        df = pd.read_csv(out_path)
        assert len(df) == len(_TRAIN_ROWS) + len(_TEST_ROWS)

    def test_split_values_and_counts_are_correct(self, raw_dir: Path, tmp_path: Path) -> None:
        out_path = tmp_path / "laudos.csv"

        build_dataset(raw_dir, out_path)

        df = pd.read_csv(out_path)
        assert set(df["split"]) == {"train", "test"}
        assert (df["split"] == "train").sum() == len(_TRAIN_ROWS)
        assert (df["split"] == "test").sum() == len(_TEST_ROWS)

    def test_label_values_are_within_allowed_set(self, raw_dir: Path, tmp_path: Path) -> None:
        out_path = tmp_path / "laudos.csv"

        build_dataset(raw_dir, out_path)

        df = pd.read_csv(out_path)
        assert set(df["label"]).issubset({"normal", "atencao", "urgente"})

    def test_ids_are_unique(self, raw_dir: Path, tmp_path: Path) -> None:
        out_path = tmp_path / "laudos.csv"

        build_dataset(raw_dir, out_path)

        df = pd.read_csv(out_path)
        assert df["id"].is_unique

    def test_condition_name_is_joined_from_labels(self, raw_dir: Path, tmp_path: Path) -> None:
        out_path = tmp_path / "laudos.csv"

        build_dataset(raw_dir, out_path)

        df = pd.read_csv(out_path)
        neoplasm_rows = df[df["condition_label"] == 1]
        assert (neoplasm_rows["condition_name"] == "neoplasms").all()

    def test_no_missing_values_in_any_column(self, raw_dir: Path, tmp_path: Path) -> None:
        out_path = tmp_path / "laudos.csv"

        build_dataset(raw_dir, out_path)

        df = pd.read_csv(out_path)
        assert not df.isna().any().any()


class TestBuildDatasetThresholds:
    """Threshold fitting/persistence and no-leakage behavior."""

    def test_thresholds_json_is_persisted_next_to_out_path(
        self, raw_dir: Path, tmp_path: Path
    ) -> None:
        out_path = tmp_path / "processed" / "laudos.csv"

        build_dataset(raw_dir, out_path, seed=7)

        thresholds_path = out_path.parent / THRESHOLDS_FILENAME
        assert thresholds_path.exists()
        loaded = load_thresholds(thresholds_path)
        assert loaded.seed == 7

    def test_thresholds_are_fit_on_train_split_only(self, raw_dir: Path, tmp_path: Path) -> None:
        """The persisted thresholds must equal fit_thresholds() on TRAIN scores alone."""
        out_path = tmp_path / "laudos.csv"

        build_dataset(raw_dir, out_path)

        df = pd.read_csv(out_path)
        train_scores = df.loc[df["split"] == "train", "urgency_score"].tolist()
        expected = fit_thresholds(train_scores)

        loaded = load_thresholds(out_path.parent / THRESHOLDS_FILENAME)
        assert loaded.low == pytest.approx(expected.low)
        assert loaded.high == pytest.approx(expected.high)

    def test_test_split_reuses_train_thresholds_without_leakage(
        self, raw_dir: Path, tmp_path: Path
    ) -> None:
        """Appending only to the test split must not change the persisted thresholds."""
        out_path = tmp_path / "laudos.csv"
        build_dataset(raw_dir, out_path)
        thresholds_before = load_thresholds(out_path.parent / THRESHOLDS_FILENAME)

        # Add extreme outlier rows to the test file only; if test scores leaked
        # into threshold fitting, the thresholds would shift.
        extra_raw_dir = tmp_path / "raw_extra"
        _write_raw_corpus(extra_raw_dir)
        extra_test = pd.read_csv(extra_raw_dir / "medical_tc_test.csv")
        outliers = pd.DataFrame(
            [
                {"condition_label": 1, "medical_abstract": "acute " * 50},
                {"condition_label": 2, "medical_abstract": "chronic " * 50},
            ]
        )
        pd.concat([extra_test, outliers], ignore_index=True).to_csv(
            extra_raw_dir / "medical_tc_test.csv", index=False
        )

        out_path_2 = tmp_path / "laudos2.csv"
        build_dataset(extra_raw_dir, out_path_2)
        thresholds_after = load_thresholds(out_path_2.parent / THRESHOLDS_FILENAME)

        assert thresholds_after.low == pytest.approx(thresholds_before.low)
        assert thresholds_after.high == pytest.approx(thresholds_before.high)


class TestBuildDatasetErrors:
    """Missing/invalid raw files raise DatasetSchemaError."""

    def test_missing_labels_file_raises(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        train_df = pd.DataFrame(_TRAIN_ROWS, columns=["condition_label", "medical_abstract"])
        train_df.to_csv(raw_dir / "medical_tc_train.csv", index=False)
        test_df = pd.DataFrame(_TEST_ROWS, columns=["condition_label", "medical_abstract"])
        test_df.to_csv(raw_dir / "medical_tc_test.csv", index=False)

        with pytest.raises(DatasetSchemaError, match="not found"):
            build_dataset(raw_dir, tmp_path / "laudos.csv")

    def test_missing_train_file_raises(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "medical_tc_labels.csv").write_text(_LABELS_CSV, encoding="utf-8")
        test_df = pd.DataFrame(_TEST_ROWS, columns=["condition_label", "medical_abstract"])
        test_df.to_csv(raw_dir / "medical_tc_test.csv", index=False)

        with pytest.raises(DatasetSchemaError, match="not found"):
            build_dataset(raw_dir, tmp_path / "laudos.csv")

    def test_unknown_condition_label_raises(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "medical_tc_labels.csv").write_text(_LABELS_CSV, encoding="utf-8")
        bad_train = pd.DataFrame([{"condition_label": 99, "medical_abstract": "Some text."}])
        bad_train.to_csv(raw_dir / "medical_tc_train.csv", index=False)
        test_df = pd.DataFrame(_TEST_ROWS, columns=["condition_label", "medical_abstract"])
        test_df.to_csv(raw_dir / "medical_tc_test.csv", index=False)

        with pytest.raises(DatasetSchemaError, match="absent from"):
            build_dataset(raw_dir, tmp_path / "laudos.csv")

    def test_labels_missing_expected_column_raises(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        pd.DataFrame([{"condition_label": 1, "wrong_col": "neoplasms"}]).to_csv(
            raw_dir / "medical_tc_labels.csv", index=False
        )
        train_df = pd.DataFrame(_TRAIN_ROWS, columns=["condition_label", "medical_abstract"])
        train_df.to_csv(raw_dir / "medical_tc_train.csv", index=False)
        test_df = pd.DataFrame(_TEST_ROWS, columns=["condition_label", "medical_abstract"])
        test_df.to_csv(raw_dir / "medical_tc_test.csv", index=False)

        with pytest.raises(DatasetSchemaError, match="missing column"):
            build_dataset(raw_dir, tmp_path / "laudos.csv")

    def test_missing_expected_column_raises(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "medical_tc_labels.csv").write_text(_LABELS_CSV, encoding="utf-8")
        pd.DataFrame([{"wrong_col": 1, "other": "x"}]).to_csv(
            raw_dir / "medical_tc_train.csv", index=False
        )
        test_df = pd.DataFrame(_TEST_ROWS, columns=["condition_label", "medical_abstract"])
        test_df.to_csv(raw_dir / "medical_tc_test.csv", index=False)

        with pytest.raises(DatasetSchemaError, match="missing column"):
            build_dataset(raw_dir, tmp_path / "laudos.csv")


class TestBuildDatasetDeterminism:
    """Running build_dataset twice must yield a byte-for-byte identical CSV."""

    def test_two_runs_produce_identical_sha256(self, raw_dir: Path, tmp_path: Path) -> None:
        out_path_1 = tmp_path / "run1" / "laudos.csv"
        out_path_2 = tmp_path / "run2" / "laudos.csv"

        build_dataset(raw_dir, out_path_1, seed=42)
        build_dataset(raw_dir, out_path_2, seed=42)

        hash_1 = hashlib.sha256(out_path_1.read_bytes()).hexdigest()
        hash_2 = hashlib.sha256(out_path_2.read_bytes()).hexdigest()
        assert hash_1 == hash_2

    def test_rerunning_over_the_same_out_path_is_also_identical(
        self, raw_dir: Path, tmp_path: Path
    ) -> None:
        out_path = tmp_path / "laudos.csv"

        build_dataset(raw_dir, out_path, seed=42)
        first_hash = hashlib.sha256(out_path.read_bytes()).hexdigest()

        build_dataset(raw_dir, out_path, seed=42)
        second_hash = hashlib.sha256(out_path.read_bytes()).hexdigest()

        assert first_hash == second_hash


def test_cli_main_builds_dataset_and_prints_path(
    monkeypatch: pytest.MonkeyPatch,
    raw_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """python -m triagem.data.build builds the dataset and prints its path."""
    out_path = tmp_path / "processed" / "laudos.csv"
    settings = Settings(raw_dir=raw_dir, data_path=out_path)
    monkeypatch.setattr("triagem.data.build.get_settings", lambda: settings)
    monkeypatch.setattr(sys, "argv", ["prog", "--seed", "42"])

    main()

    captured = capsys.readouterr()
    assert str(out_path) in captured.out
    assert out_path.exists()


def test_cli_main_accepts_explicit_raw_dir_and_out_path(
    monkeypatch: pytest.MonkeyPatch,
    raw_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--raw-dir and --out-path override the settings-derived defaults."""
    out_path = tmp_path / "custom" / "laudos.csv"
    monkeypatch.setattr("triagem.data.build.get_settings", lambda: Settings())
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--raw-dir", str(raw_dir), "--out-path", str(out_path), "--seed", "1"],
    )

    main()

    assert out_path.exists()


@pytest.mark.network
def test_real_pipeline_meets_volume_and_class_balance_floors(tmp_path: Path) -> None:
    """End-to-end on the real corpus: >= 2000 rows and no class below 15%.

    Deselected by default (pyproject.toml addopts excludes the ``network``
    marker). Run explicitly with:
    ``env -u VIRTUAL_ENV poetry run pytest -m network tests/data/test_build.py``
    """
    raw_dir = tmp_path / "raw"
    out_path = tmp_path / "processed" / "laudos.csv"

    download_raw(dest_dir=raw_dir, timeout=60)
    build_dataset(raw_dir, out_path, seed=42)

    df = pd.read_csv(out_path)
    assert len(df) >= 2000, "PDF requires a minimum of 2000 samples"

    total = len(df)
    for label in ("normal", "atencao", "urgente"):
        share = (df["label"] == label).sum() / total
        assert share >= 0.15, f"{label} share {share:.3f} is below the 15% floor"
