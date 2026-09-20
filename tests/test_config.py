"""Tests for triagem.config (Settings, get_settings) and triagem.logging_conf."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from triagem.config import Settings, get_settings
from triagem.logging_conf import setup_logging

_ENV_KEYS = [
    "TRIAGEM_DATA_DIR",
    "TRIAGEM_RAW_DIR",
    "TRIAGEM_PROCESSED_DIR",
    "TRIAGEM_MODELS_DIR",
    "TRIAGEM_DATA_PATH",
    "TRIAGEM_MODEL_BACKEND",
    "TRIAGEM_SEED",
    "TRIAGEM_API_PORT",
    "TRIAGEM_LOG_LEVEL",
    "TRIAGEM_DATASET_BASE_URL",
]


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run every test with no TRIAGEM_* env vars and no local .env file."""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_are_functional_without_env() -> None:
    """Settings() must work with zero configuration (no .env, no env vars)."""
    settings = Settings()

    assert settings.data_dir == Path("data")
    assert settings.raw_dir == Path("data/raw")
    assert settings.processed_dir == Path("data/processed")
    assert settings.models_dir == Path("models")
    assert settings.data_path == Path("data/processed/laudos.csv")
    assert settings.model_backend == "onnx"  # T22: onnx by default, sklearn is the fallback
    assert settings.seed == 42
    assert settings.api_port == 8000
    assert settings.log_level == "INFO"
    assert settings.dataset_base_url.startswith("https://raw.githubusercontent.com/")


def test_every_field_is_overridable_via_triagem_prefixed_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every documented field must be overridable via TRIAGEM_<FIELD>."""
    monkeypatch.setenv("TRIAGEM_DATA_DIR", "custom_data")
    monkeypatch.setenv("TRIAGEM_RAW_DIR", "custom_data/raw")
    monkeypatch.setenv("TRIAGEM_PROCESSED_DIR", "custom_data/processed")
    monkeypatch.setenv("TRIAGEM_MODELS_DIR", "custom_models")
    monkeypatch.setenv("TRIAGEM_DATA_PATH", "custom_data/processed/laudos.csv")
    monkeypatch.setenv("TRIAGEM_MODEL_BACKEND", "sklearn")
    monkeypatch.setenv("TRIAGEM_SEED", "7")
    monkeypatch.setenv("TRIAGEM_API_PORT", "9001")
    monkeypatch.setenv("TRIAGEM_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("TRIAGEM_DATASET_BASE_URL", "https://example.invalid/corpus/")

    settings = Settings()

    assert settings.data_dir == Path("custom_data")
    assert settings.raw_dir == Path("custom_data/raw")
    assert settings.processed_dir == Path("custom_data/processed")
    assert settings.models_dir == Path("custom_models")
    assert settings.data_path == Path("custom_data/processed/laudos.csv")
    assert settings.model_backend == "sklearn"
    assert settings.seed == 7
    assert settings.api_port == 9001
    assert settings.log_level == "DEBUG"
    assert settings.dataset_base_url == "https://example.invalid/corpus/"


def test_unprefixed_env_vars_are_ignored() -> None:
    """Keys without the TRIAGEM_ prefix must never leak into Settings."""
    os.environ["API_PORT"] = "1"
    try:
        settings = Settings()
        assert settings.api_port == 8000
    finally:
        del os.environ["API_PORT"]


def test_invalid_model_backend_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """model_backend is a closed set: 'sklearn' or 'onnx', nothing else."""
    monkeypatch.setenv("TRIAGEM_MODEL_BACKEND", "tensorflow")

    with pytest.raises(ValidationError):
        Settings()


def test_get_settings_is_cached() -> None:
    """get_settings() must return the same instance across calls (lru_cache)."""
    first = get_settings()
    second = get_settings()

    assert first is second


def test_get_settings_reflects_env_after_cache_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    """cache_clear() lets tests/tools force a fresh read of the environment."""
    first = get_settings()

    monkeypatch.setenv("TRIAGEM_API_PORT", "9500")
    get_settings.cache_clear()
    second = get_settings()

    assert first.api_port == 8000
    assert second.api_port == 9500


@pytest.fixture
def _restore_root_logging() -> Iterator[None]:
    """Snapshot/restore root logger state so tests don't leak handlers."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    root.handlers.clear()
    root.handlers.extend(original_handlers)
    root.setLevel(original_level)


def test_setup_logging_emits_single_line_json_with_required_fields(
    capsys: pytest.CaptureFixture[str],
    _restore_root_logging: None,
) -> None:
    """Each event is one JSON line with level/logger/message fields."""
    setup_logging(level="INFO")
    logger = logging.getLogger("triagem.test")
    logger.info("hello world")

    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if line.strip()]

    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["level"] == "INFO"
    assert payload["logger"] == "triagem.test"
    assert payload["message"] == "hello world"


def test_setup_logging_defaults_to_settings_log_level(
    monkeypatch: pytest.MonkeyPatch,
    _restore_root_logging: None,
) -> None:
    """With no explicit level, setup_logging() falls back to Settings.log_level."""
    monkeypatch.setenv("TRIAGEM_LOG_LEVEL", "WARNING")
    get_settings.cache_clear()

    setup_logging()

    assert logging.getLogger().level == logging.WARNING


def test_setup_logging_filters_below_configured_level(
    capsys: pytest.CaptureFixture[str],
    _restore_root_logging: None,
) -> None:
    """A DEBUG record must be dropped when the configured level is INFO."""
    setup_logging(level="INFO")
    logger = logging.getLogger("triagem.test.debug")
    logger.debug("should not appear")

    captured = capsys.readouterr()
    assert captured.err == ""


def test_setup_logging_includes_exc_info_when_logging_an_exception(
    capsys: pytest.CaptureFixture[str],
    _restore_root_logging: None,
) -> None:
    """Exceptions logged via logger.exception() carry an exc_info field."""
    setup_logging(level="INFO")
    logger = logging.getLogger("triagem.test.exc")

    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("failed")

    captured = capsys.readouterr()
    payload = json.loads(captured.err.strip())
    assert payload["message"] == "failed"
    assert "ValueError: boom" in payload["exc_info"]
