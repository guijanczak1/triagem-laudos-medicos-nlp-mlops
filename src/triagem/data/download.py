"""Reproducible, credential-free download of the Medical Abstracts TC Corpus.

Downloads the three public CSV files published by the
``sebischair/Medical-Abstracts-TC-Corpus`` project (CC BY-SA 3.0) directly
from GitHub's raw-content endpoint (no API token, no authentication),
validates their schema, and records the dataset's license/attribution
alongside the downloaded files so redistribution stays compliant.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import requests

from triagem.config import Settings, get_settings
from triagem.exceptions import DatasetDownloadError

logger = logging.getLogger(__name__)

# Logical name -> upstream filename. Order matters only for readability; the
# download loop below is independent per file.
_FILES_BY_KEY: dict[str, str] = {
    "train": "medical_tc_train.csv",
    "test": "medical_tc_test.csv",
    "labels": "medical_tc_labels.csv",
}

# Columns each file must expose in its header. The corpus is redistributed as
# plain CSV with a header row; if upstream ever changes shape, downloads must
# fail loudly (DatasetDownloadError) rather than silently produce a dataset
# the rest of the pipeline would misinterpret.
_EXPECTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "medical_tc_train.csv": ("condition_label", "medical_abstract"),
    "medical_tc_test.csv": ("condition_label", "medical_abstract"),
    "medical_tc_labels.csv": ("condition_label", "condition_name"),
}

LICENSE_FILENAME = "LICENSE_DATASET.md"

_LICENSE_TEXT = """# Dataset license and attribution

Source: [sebischair/Medical-Abstracts-TC-Corpus](\
https://github.com/sebischair/Medical-Abstracts-TC-Corpus)

License: **CC BY-SA 3.0** (Creative Commons Attribution-ShareAlike 3.0 Unported)

This directory contains an unmodified copy of the corpus CSV files
(`medical_tc_train.csv`, `medical_tc_test.csv`, `medical_tc_labels.csv`),
downloaded as-is from the project's GitHub repository (`main` branch, raw
content). Redistribution and reuse are permitted under CC BY-SA 3.0 provided
that appropriate credit is given to the original authors and that any
derivative work is shared under the same license.

Label mapping (`medical_tc_labels.csv`): 1 = neoplasms, 2 = digestive system
diseases, 3 = nervous system diseases, 4 = cardiovascular diseases,
5 = general pathological conditions.
"""


def _validate_columns(filename: str, header_line: str) -> None:
    """Raise if ``header_line`` is missing a column required for ``filename``.

    Args:
        filename: Name of the CSV being validated (used in the error message
            and to look up the expected columns).
        header_line: Raw first line of the downloaded CSV content.

    Raises:
        DatasetDownloadError: If any expected column is absent from the
            header, with an actionable message naming both the expected and
            the actual columns found.
    """
    expected = _EXPECTED_COLUMNS[filename]
    actual_columns = [col.strip() for col in header_line.strip().split(",")]
    missing = [col for col in expected if col not in actual_columns]
    if missing:
        raise DatasetDownloadError(
            f"{filename}: unexpected schema, missing column(s) {missing}. "
            f"Expected the header to contain {list(expected)}, got {actual_columns}. "
            "The upstream dataset may have changed its format; if this is "
            "intentional, update triagem.data.download._EXPECTED_COLUMNS."
        )


def _write_license_file(dest_dir: Path) -> Path:
    """Write the CC BY-SA 3.0 attribution file into ``dest_dir``.

    Args:
        dest_dir: Directory the raw corpus files were downloaded into.

    Returns:
        Path to the written license/attribution file.
    """
    license_path = dest_dir / LICENSE_FILENAME
    license_path.write_text(_LICENSE_TEXT, encoding="utf-8")
    return license_path


def download_raw(
    dest_dir: Path | None = None,
    force: bool = False,
    timeout: int = 60,
    settings: Settings | None = None,
) -> dict[str, Path]:
    """Download the Medical Abstracts TC Corpus CSVs to ``dest_dir``.

    Fetches ``medical_tc_train.csv``, ``medical_tc_test.csv`` and
    ``medical_tc_labels.csv`` from the public, credential-free GitHub raw
    endpoint (``Settings.dataset_base_url``), validates the header of each
    file against its expected schema, and writes a CC BY-SA 3.0
    license/attribution file alongside them.

    Args:
        dest_dir: Directory the CSVs (and the license file) are written to.
            Defaults to ``settings.raw_dir``.
        force: When ``False`` (default), a file already present on disk is
            left untouched and not re-downloaded (idempotent). When ``True``,
            every file is re-downloaded unconditionally.
        timeout: Per-request timeout in seconds passed to ``requests.get``.
        settings: Optional pre-resolved ``Settings`` instance, mainly for
            tests; defaults to ``get_settings()``.

    Returns:
        A mapping of logical name (``"train"``, ``"test"``, ``"labels"``,
        ``"license"``) to the corresponding local ``Path``.

    Raises:
        DatasetDownloadError: If a request fails or times out, the server
            returns a non-2xx status, the response body is empty, or the
            downloaded content does not match the expected CSV schema.
    """
    resolved_settings = settings or get_settings()
    resolved_dest_dir = dest_dir if dest_dir is not None else resolved_settings.raw_dir
    resolved_dest_dir.mkdir(parents=True, exist_ok=True)

    base_url = resolved_settings.dataset_base_url.rstrip("/") + "/"
    result: dict[str, Path] = {}

    for key, filename in _FILES_BY_KEY.items():
        dest_path = resolved_dest_dir / filename

        if dest_path.exists() and not force:
            logger.info("%s already present at %s, skipping download", filename, dest_path)
            result[key] = dest_path
            continue

        url = base_url + filename
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DatasetDownloadError(
                f"failed to download {filename} from {url}: {exc}. "
                "This is a public, credential-free endpoint - check network "
                "connectivity and that the upstream repository/branch still exists."
            ) from exc

        content = response.text
        if not content.strip():
            raise DatasetDownloadError(f"{filename}: downloaded content from {url} is empty.")

        header_line = content.splitlines()[0]
        _validate_columns(filename, header_line)

        dest_path.write_text(content, encoding="utf-8")
        logger.info("downloaded %s to %s (%d bytes)", filename, dest_path, len(content))
        result[key] = dest_path

    result["license"] = _write_license_file(resolved_dest_dir)
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``python -m triagem.data.download``."""
    parser = argparse.ArgumentParser(
        description=(
            "Download the Medical Abstracts TC Corpus (CC BY-SA 3.0) from "
            "GitHub raw content. No credentials required."
        )
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=None,
        help="Destination directory (defaults to Settings.raw_dir, i.e. data/raw).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds (default: 60).",
    )
    return parser


def main() -> None:
    """CLI entry point: ``python -m triagem.data.download``."""
    from triagem.logging_conf import setup_logging

    setup_logging()
    args = _build_arg_parser().parse_args()
    paths = download_raw(dest_dir=args.dest_dir, force=args.force, timeout=args.timeout)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
