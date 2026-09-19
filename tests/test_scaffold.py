"""Smoke tests for the project scaffold (packaging, versioning)."""

from triagem import __version__


def test_package_version_is_a_non_empty_string() -> None:
    """The installed package must expose a semantic version string."""
    assert isinstance(__version__, str)
    assert __version__
