"""Tests for triagem.data.download.

Unit tests mock every HTTP call via ``monkeypatch`` (no network, no
credentials). The single real-network check is marked ``@pytest.mark.network``
and deselected by default (see ``addopts`` in pyproject.toml); run it
explicitly with ``poetry run pytest -m network tests/data/test_download.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import requests

from triagem.config import Settings
from triagem.data.download import LICENSE_FILENAME, download_raw, main
from triagem.exceptions import DatasetDownloadError

_TRAIN_CSV = 'condition_label,medical_abstract\n1,"Patient presents with malignant neoplasm."\n'
_TEST_CSV = 'condition_label,medical_abstract\n4,"Chronic stable cardiovascular condition."\n'
_LABELS_CSV = (
    "condition_label,condition_name\n"
    "1,neoplasms\n"
    "2,digestive system diseases\n"
    "3,nervous system diseases\n"
    "4,cardiovascular diseases\n"
    "5,general pathological conditions\n"
)

_CONTENT_BY_FILENAME = {
    "medical_tc_train.csv": _TRAIN_CSV,
    "medical_tc_test.csv": _TEST_CSV,
    "medical_tc_labels.csv": _LABELS_CSV,
}


class _FakeResponse:
    """Minimal stand-in for ``requests.Response`` used to mock network I/O."""

    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        """Mimic ``requests.Response.raise_for_status`` for status >= 400."""
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """A Settings instance isolated to a temp directory, with no .env lookup."""
    return Settings(
        raw_dir=tmp_path / "raw",
        dataset_base_url="https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/",
    )


def _install_success_mock(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch requests.get with a fake handler; return the list of requested URLs."""
    requested: list[str] = []

    def _fake_get(url: str, timeout: int) -> _FakeResponse:
        requested.append(url)
        filename = url.rsplit("/", 1)[-1]
        return _FakeResponse(_CONTENT_BY_FILENAME[filename])

    monkeypatch.setattr(requests, "get", _fake_get)
    return requested


def test_download_raw_downloads_all_three_files_and_license(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """All three CSVs and the license/attribution file are written to disk."""
    requested = _install_success_mock(monkeypatch)

    result = download_raw(settings=settings)

    assert set(result) == {"train", "test", "labels", "license"}
    assert result["train"].read_text(encoding="utf-8") == _TRAIN_CSV
    assert result["test"].read_text(encoding="utf-8") == _TEST_CSV
    assert result["labels"].read_text(encoding="utf-8") == _LABELS_CSV
    assert result["license"].name == LICENSE_FILENAME
    assert "CC BY-SA 3.0" in result["license"].read_text(encoding="utf-8")
    assert len(requested) == 3


def test_download_raw_creates_dest_dir(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    """The destination directory is created if it does not already exist."""
    _install_success_mock(monkeypatch)
    assert not settings.raw_dir.exists()

    download_raw(settings=settings)

    assert settings.raw_dir.is_dir()


def test_download_raw_is_idempotent_by_default(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """A second call with force=False must not re-download existing files."""
    requested = _install_success_mock(monkeypatch)
    download_raw(settings=settings)
    assert len(requested) == 3

    download_raw(settings=settings)

    assert len(requested) == 3


def test_download_raw_force_redownloads(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """force=True re-downloads every file even if already present on disk."""
    requested = _install_success_mock(monkeypatch)
    download_raw(settings=settings)
    assert len(requested) == 3

    download_raw(settings=settings, force=True)

    assert len(requested) == 6


def test_download_raw_explicit_dest_dir_overrides_settings(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    """An explicit dest_dir argument takes precedence over Settings.raw_dir."""
    _install_success_mock(monkeypatch)
    other_dir = tmp_path / "elsewhere"

    result = download_raw(dest_dir=other_dir, settings=settings)

    assert result["train"].parent == other_dir
    assert not settings.raw_dir.exists()


def test_download_raw_raises_on_schema_mismatch(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """A header missing an expected column raises DatasetDownloadError."""

    def _fake_get(url: str, timeout: int) -> _FakeResponse:
        return _FakeResponse("wrong_col,other_col\n1,x\n")

    monkeypatch.setattr(requests, "get", _fake_get)

    with pytest.raises(DatasetDownloadError, match="missing column"):
        download_raw(settings=settings)


def test_download_raw_raises_on_empty_response(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """An empty response body raises DatasetDownloadError."""

    def _fake_get(url: str, timeout: int) -> _FakeResponse:
        return _FakeResponse("   ")

    monkeypatch.setattr(requests, "get", _fake_get)

    with pytest.raises(DatasetDownloadError, match="empty"):
        download_raw(settings=settings)


def test_download_raw_raises_on_http_error(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """A non-2xx HTTP status raises DatasetDownloadError with an actionable message."""

    def _fake_get(url: str, timeout: int) -> _FakeResponse:
        return _FakeResponse("x", status_code=404)

    monkeypatch.setattr(requests, "get", _fake_get)

    with pytest.raises(DatasetDownloadError, match="failed to download"):
        download_raw(settings=settings)


def test_download_raw_raises_on_connection_error(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """A network-level failure (no connectivity) raises DatasetDownloadError."""

    def _raise_connection_error(url: str, timeout: int) -> _FakeResponse:
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(requests, "get", _raise_connection_error)

    with pytest.raises(DatasetDownloadError, match="failed to download"):
        download_raw(settings=settings)


def test_labels_csv_schema_uses_condition_name_column(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """medical_tc_labels.csv is validated against its own (different) schema."""
    _install_success_mock(monkeypatch)

    result = download_raw(settings=settings)

    header = result["labels"].read_text(encoding="utf-8").splitlines()[0]
    assert header == "condition_label,condition_name"


def test_cli_main_downloads_and_prints_paths(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """python -m triagem.data.download downloads files and prints their paths."""
    _install_success_mock(monkeypatch)
    monkeypatch.setattr("triagem.data.download.get_settings", lambda: settings)
    monkeypatch.setattr(sys, "argv", ["prog"])

    main()

    captured = capsys.readouterr()
    assert "train:" in captured.out
    assert "license:" in captured.out


@pytest.mark.network
def test_download_raw_real_network_download(tmp_path: Path) -> None:
    """Integration check: downloads the real corpus from GitHub raw content.

    Deselected by default (pyproject.toml addopts excludes the ``network``
    marker). Run explicitly with:
    ``env -u VIRTUAL_ENV poetry run pytest -m network tests/data/test_download.py``
    """
    real_settings = Settings(raw_dir=tmp_path / "raw")

    result = download_raw(settings=real_settings, timeout=60)

    for key in ("train", "test", "labels"):
        assert result[key].exists()
        assert result[key].stat().st_size > 0
    header = result["labels"].read_text(encoding="utf-8").splitlines()[0]
    assert header == "condition_label,condition_name"
