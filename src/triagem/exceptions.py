"""Shared exception types used across the triagem package.

Kept in one small, dependency-free module so every layer (data ingestion,
dataset loading, model serving) can raise and catch the same well-known
error types without introducing import cycles.
"""

from __future__ import annotations


class TriagemError(Exception):
    """Base class for all triagem-specific errors."""


class DatasetDownloadError(TriagemError):
    """Raised when the raw dataset cannot be downloaded or fails validation."""


class DatasetSchemaError(TriagemError):
    """Raised when a loaded dataset does not match the expected schema."""


class ModelArtifactNotFound(TriagemError):
    """Raised when a required model artifact is missing from disk."""
