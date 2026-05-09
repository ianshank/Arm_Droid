"""Integration tests for factory dispatch by config.

``build_arm_driver`` selects the correct concrete driver class based on
``cfg.arm_driver_kind`` (or, falling back, the legacy ``cfg.mock_hardware``
bool), and the type returned satisfies the protocol. Also verifies that
the orchestrator wiring does not double-build the driver (only one driver
instance is shared between the controller's ActionPrimitives and the
orchestrator's stored reference).
"""

from __future__ import annotations

import importlib.util
import warnings

import pytest

from armdroid.config.schema import ArmSettings
from armdroid.domain.protocols import ArmDriverProtocol
from armdroid.hardware.mock_arm_driver import MockArmDriver
from armdroid.orchestration.factory import build_arm_driver, build_arm_orchestrator

# Esp32JsonDriver requires pyserial at construction time. Skip the
# real-hardware factory dispatch test when pyserial is not installed
# rather than failing with ArmDriverError.
_HAS_PYSERIAL = importlib.util.find_spec("serial") is not None


class TestFactoryDispatch:
    def test_arm_driver_kind_mock_returns_mock(self) -> None:
        """Modern path: arm_driver_kind='mock' resolves to MockArmDriver."""
        cfg = ArmSettings(arm_driver_kind="mock")
        drv = build_arm_driver(cfg)
        assert isinstance(drv, MockArmDriver)
        assert isinstance(drv, ArmDriverProtocol)

    def test_legacy_mock_hardware_true_resolves_to_mock(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Legacy path: mock_hardware=True (no arm_driver_kind) → MockArmDriver."""
        # Override the autouse env var so the legacy fall-through fires.
        monkeypatch.delenv("ARMDROID_ARM_DRIVER_KIND", raising=False)
        monkeypatch.setenv("ARMDROID_SUPPRESS_DEPRECATION", "1")
        cfg = ArmSettings(mock_hardware=True, arm_driver_kind=None)
        drv = build_arm_driver(cfg)
        assert isinstance(drv, MockArmDriver)
        assert isinstance(drv, ArmDriverProtocol)

    @pytest.mark.skipif(
        not _HAS_PYSERIAL,
        reason="pyserial not installed — Esp32JsonDriver construction requires it",
    )
    def test_arm_driver_kind_esp32_returns_esp32(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Modern path: arm_driver_kind='esp32' resolves to Esp32JsonDriver.

        The driver is instantiated but not connected, so no actual port is
        opened — we only verify the type. Skipped when pyserial is missing
        because the driver's ``__init__`` raises ``ArmDriverError`` in that
        case.
        """
        monkeypatch.delenv("ARMDROID_ARM_DRIVER_KIND", raising=False)
        cfg = ArmSettings(arm_driver_kind="esp32")
        drv = build_arm_driver(cfg)
        # Avoid importing the concrete class at module top so this file
        # can run on systems without pyserial installed (the import in
        # esp32_json_driver guards against that).
        assert drv.__class__.__name__ == "Esp32JsonDriver"
        assert isinstance(drv, ArmDriverProtocol)

    @pytest.mark.skipif(
        not _HAS_PYSERIAL,
        reason="pyserial not installed — Esp32JsonDriver construction requires it",
    )
    def test_legacy_mock_hardware_false_resolves_to_esp32(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Legacy path: mock_hardware=False (no arm_driver_kind) → Esp32JsonDriver."""
        monkeypatch.delenv("ARMDROID_ARM_DRIVER_KIND", raising=False)
        cfg = ArmSettings(mock_hardware=False, arm_driver_kind=None)
        drv = build_arm_driver(cfg)
        assert drv.__class__.__name__ == "Esp32JsonDriver"
        assert isinstance(drv, ArmDriverProtocol)

    def test_legacy_mock_hardware_true_emits_deprecation_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Setting mock_hardware=True without arm_driver_kind warns once.

        Documents — and protects — the deprecation path until v0.4.0 removes
        the legacy bool entirely.
        """
        from armdroid.orchestration._driver_kind import _reset_warned_for_tests

        _reset_warned_for_tests()
        monkeypatch.delenv("ARMDROID_ARM_DRIVER_KIND", raising=False)
        monkeypatch.delenv("ARMDROID_SUPPRESS_DEPRECATION", raising=False)
        cfg = ArmSettings(mock_hardware=True, arm_driver_kind=None)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            build_arm_driver(cfg)
        deprecation_warnings = [w for w in captured if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1


class TestOrchestratorDriverSharing:
    def test_one_driver_shared_between_orch_and_controller(self) -> None:
        cfg = ArmSettings(arm_driver_kind="mock")
        orch = build_arm_orchestrator(cfg)
        # The orchestrator's driver and the controller's primitives' driver
        # are the same object — built once and shared, so we don't open
        # two serial connections to the same hardware.
        assert orch.driver is orch.controller.primitives.driver
