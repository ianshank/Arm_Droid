"""Shared test fixtures for armdroid test suite."""

from __future__ import annotations

import pytest

from armdroid.config.schema import ArmSettings


@pytest.fixture(autouse=True)
def _mock_arm_driver_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a mock driver in tests; isolate from developer shell leakage.

    Migrated from ``ARMDROID_MOCK_HARDWARE=true`` (peer-review C2/N3): the
    new ``arm_driver_kind`` field is the source of truth, and exercising
    it in CI keeps the new code path covered. Tests that explicitly want
    to exercise the legacy bool path construct ``ArmSettings(mock_hardware=...)``
    directly and use ``pytest.warns(DeprecationWarning)`` to assert the
    deprecation surfaces.

    The ``delenv`` calls neutralise developer-shell leakage that would
    otherwise contradict explicit ``ArmSettings(...)`` constructions.
    """
    monkeypatch.setenv("ARMDROID_ARM_DRIVER_KIND", "mock")
    monkeypatch.delenv("ARMDROID_MOCK_HARDWARE", raising=False)


@pytest.fixture
def mock_settings() -> ArmSettings:
    """Return an ArmSettings instance with mock hardware enabled.

    Uses the new ``arm_driver_kind`` field. The legacy ``mock_hardware``
    bool is intentionally left at its default ``False`` so the
    deprecation warning is not triggered in tests using this fixture.
    """
    return ArmSettings(arm_driver_kind="mock")
