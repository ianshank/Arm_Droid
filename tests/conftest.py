"""Shared test fixtures for armdroid test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _mock_hardware_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure mock hardware is enabled for all tests."""
    monkeypatch.setenv("ARMDROID_MOCK_HARDWARE", "true")
