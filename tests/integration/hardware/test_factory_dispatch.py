"""Integration tests for factory dispatch by config.

`build_arm_driver` selects the correct concrete driver class based on
`cfg.mock_hardware`, and the type returned satisfies the protocol. Also
verifies that the orchestrator wiring does not double-build the driver
(only one driver instance is shared between the controller's
ActionPrimitives and the orchestrator's stored reference).
"""

from __future__ import annotations

import importlib.util

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
    def test_mock_hardware_returns_mock(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        drv = build_arm_driver(cfg)
        assert isinstance(drv, MockArmDriver)
        assert isinstance(drv, ArmDriverProtocol)

    @pytest.mark.skipif(
        not _HAS_PYSERIAL,
        reason="pyserial not installed — Esp32JsonDriver construction requires it",
    )
    def test_real_hardware_returns_esp32(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When mock_hardware=False, the factory constructs Esp32JsonDriver.

        The driver is instantiated but not connected, so no actual port is
        opened — we only verify the type. Skipped when pyserial is missing
        because the driver's ``__init__`` raises ``ArmDriverError`` in that
        case.
        """
        cfg = ArmSettings(mock_hardware=False)
        drv = build_arm_driver(cfg)
        # Avoid importing the concrete class at module top so this file
        # can run on systems without pyserial installed (the import in
        # esp32_json_driver guards against that).
        assert drv.__class__.__name__ == "Esp32JsonDriver"
        assert isinstance(drv, ArmDriverProtocol)


class TestOrchestratorDriverSharing:
    def test_one_driver_shared_between_orch_and_controller(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        orch = build_arm_orchestrator(cfg)
        # The orchestrator's driver and the controller's primitives' driver
        # are the same object — built once and shared, so we don't open
        # two serial connections to the same hardware.
        assert orch.driver is orch.controller.primitives.driver
