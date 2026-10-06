"""Pytest fixtures shared by all tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from handwriting.sample_store import ProfileStore


@pytest.fixture
def store(tmp_path: Path) -> ProfileStore:
    """A profile store in a fresh temporary data directory."""
    return ProfileStore(tmp_path / "data")
