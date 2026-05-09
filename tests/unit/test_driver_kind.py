"""Tests for resolve_driver_kind() precedence + deprecation semantics."""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import pytest

from armdroid.config.schema import ArmSettings

# Imports of underscore-prefixed symbols are explicit (not via __all__).
from armdroid.orchestration._driver_kind import (
    _DEPRECATION_MSG,
    _reset_warned_for_tests,
    resolve_driver_kind,
)


@pytest.fixture(autouse=True)
def _reset_warned() -> Iterator[None]:
    """Reset once-per-process flag at start AND end of each test."""
    _reset_warned_for_tests()
    yield
    _reset_warned_for_tests()


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise developer-shell env-var leakage that would skew tests."""
    # The session-wide conftest already sets ARMDROID_ARM_DRIVER_KIND=mock;
    # individual tests in this file need to control these vars themselves.
    monkeypatch.delenv("ARMDROID_ARM_DRIVER_KIND", raising=False)
    monkeypatch.delenv("ARMDROID_MOCK_HARDWARE", raising=False)
    monkeypatch.delenv("ARMDROID_SUPPRESS_DEPRECATION", raising=False)


class TestResolvePrecedence:
    def test_explicit_kind_wins_over_legacy_bool(self) -> None:
        cfg = ArmSettings(mock_hardware=True, arm_driver_kind="esp32")
        assert resolve_driver_kind(cfg) == "esp32"

    def test_explicit_mock_kind_skips_deprecation_path(self) -> None:
        cfg = ArmSettings(arm_driver_kind="mock")
        assert resolve_driver_kind(cfg) == "mock"

    def test_legacy_mock_true_resolves_to_mock(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ARMDROID_SUPPRESS_DEPRECATION", "1")
        cfg = ArmSettings(mock_hardware=True, arm_driver_kind=None)
        assert resolve_driver_kind(cfg) == "mock"

    def test_legacy_mock_false_resolves_to_esp32(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ARMDROID_SUPPRESS_DEPRECATION", "1")
        cfg = ArmSettings(mock_hardware=False, arm_driver_kind=None)
        assert resolve_driver_kind(cfg) == "esp32"

    def test_env_var_override_via_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ARMDROID_ARM_DRIVER_KIND", "esp32")
        cfg = ArmSettings()
        # pydantic_settings populates arm_driver_kind from env var
        assert cfg.arm_driver_kind == "esp32"
        assert resolve_driver_kind(cfg) == "esp32"


class TestDeprecation:
    def test_emits_once_per_process(self) -> None:
        cfg = ArmSettings(mock_hardware=True, arm_driver_kind=None)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            resolve_driver_kind(cfg)
            resolve_driver_kind(cfg)
            resolve_driver_kind(cfg)
        deprecation_warnings = [w for w in captured if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1
        assert _DEPRECATION_MSG in str(deprecation_warnings[0].message)

    def test_suppression_env_var_silences_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ARMDROID_SUPPRESS_DEPRECATION", "1")
        cfg = ArmSettings(mock_hardware=True, arm_driver_kind=None)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            resolve_driver_kind(cfg)
        assert not any(issubclass(w.category, DeprecationWarning) for w in captured)

    def test_explicit_kind_skips_deprecation(self) -> None:
        cfg = ArmSettings(arm_driver_kind="mock")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            resolve_driver_kind(cfg)
        assert not any(issubclass(w.category, DeprecationWarning) for w in captured)

    def test_mock_false_with_no_explicit_kind_does_not_warn(self) -> None:
        # Default settings: mock_hardware defaults to False; no legacy
        # path is exercised, so no deprecation should fire.
        cfg = ArmSettings(mock_hardware=False, arm_driver_kind=None)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            resolve_driver_kind(cfg)
        assert not any(issubclass(w.category, DeprecationWarning) for w in captured)
