"""Centralized, environment-overridable application configuration.

All settings are exposed through :class:`Settings` (pydantic-settings) and
resolved via :func:`get_settings`, which caches a single instance for the
process. Every field can be overridden by an environment variable prefixed
with ``TRIAGEM_`` (e.g. ``TRIAGEM_API_PORT=9000``); with no environment or
``.env`` file present, all defaults are functional on their own.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ModelBackend = Literal["sklearn", "onnx"]


class Settings(BaseSettings):
    """Application settings loaded from the environment and/or ``.env``.

    Attributes:
        data_dir: Root directory for all dataset files.
        raw_dir: Destination of the raw, downloaded corpus.
        processed_dir: Destination of the processed dataset and derived
            artifacts (e.g. urgency thresholds).
        models_dir: Destination of trained/exported model artifacts.
        data_path: Path to the processed dataset consumed by
            ``load_dataset()``. Pointing this at another CSV with the same
            columns swaps the dataset without touching code.
        model_backend: Inference backend used by ``Predictor``. Defaults to
            ``"onnx"`` (T21's benchmark measured it ~69% faster than
            ``sklearn`` at p50). If ``models/model.onnx`` is missing at API
            startup, ``triagem.serving.api`` falls back to ``"sklearn"``
            explicitly -- logged as a WARNING, never silent -- rather than
            failing to start; see task T22.
        seed: Random seed used across data splitting, training and
            benchmarking, for reproducibility.
        api_port: Port the uvicorn server binds to.
        log_level: Root logging level (e.g. ``DEBUG``, ``INFO``, ``WARNING``).
        dataset_base_url: Base URL used to download the raw Medical
            Abstracts TC Corpus.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRIAGEM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    data_dir: Path = Field(default=Path("data"))
    raw_dir: Path = Field(default=Path("data/raw"))
    processed_dir: Path = Field(default=Path("data/processed"))
    models_dir: Path = Field(default=Path("models"))
    data_path: Path = Field(default=Path("data/processed/laudos.csv"))
    model_backend: ModelBackend = Field(default="onnx")
    seed: int = Field(default=42)
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    dataset_base_url: str = Field(
        default="https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance.

    Cached with ``lru_cache`` so the whole application shares one instance
    and environment parsing happens once. Tests that need to observe a
    changed environment must call ``get_settings.cache_clear()`` first.
    """
    return Settings()
