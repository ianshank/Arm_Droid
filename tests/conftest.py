"""Shared test fixtures for armdroid test suite."""

from __future__ import annotations

import pytest

from armdroid.config.schema import ArmSettings


@pytest.fixture(autouse=True)
def _mock_hardware_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure mock hardware is enabled for all tests."""
    monkeypatch.setenv("ARMDROID_MOCK_HARDWARE", "true")


@pytest.fixture
def mock_settings() -> ArmSettings:
    """Return an ArmSettings instance with mock hardware enabled."""
    return ArmSettings(mock_hardware=True)
