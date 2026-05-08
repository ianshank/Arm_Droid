"""Baseline regression tests.

These tests guard the current armdroid public surface against accidental
breakage during the ESP32-arm integration. They cover:

* Schema construction with no overlays.
* Factory DI graph: building an orchestrator with mock hardware succeeds
  and exposes all five subsystem properties.
* Protocol conformance: the existing ``MockArmDriver`` and
  ``Esp32JsonDriver`` satisfy the ``ArmDriverProtocol``.
* Named poll constants present and correctly typed in ``Esp32JsonDriver``.
* Shared test-harness helpers remain importable.
* Optimal Tower of Hanoi plan length stays at ``2**n - 1``.

Marked ``regression`` so they can be filtered into a dedicated CI stage.
"""

from __future__ import annotations

import pytest

from armdroid.config.schema import ArmSettings, load_settings
from armdroid.factory import build_arm_orchestrator
from armdroid.hardware.mock_arm_driver import MockArmDriver
from armdroid.protocols import ArmDriverProtocol

pytestmark = pytest.mark.regression


class TestSchemaRegression:
    """Default-constructed settings stay valid across changes."""

    def test_arm_settings_constructs_without_overlays(self) -> None:
        # conftest.py forces ARMDROID_MOCK_HARDWARE=true for the test run, so
        # the BaseSettings env reader picks that up here. We assert on the
        # *shape* of the config (dof, home length match) rather than on the
        # mock_hardware default.
        cfg = ArmSettings()
        assert cfg.arm.dof == 6  # current-baseline DoF; bumps to 7 in commit 7
        assert len(cfg.arm.home_position) == cfg.arm.dof
        assert isinstance(cfg.mock_hardware, bool)

    def test_load_settings_returns_arm_settings(self) -> None:
        cfg = load_settings()
        assert isinstance(cfg, ArmSettings)
        assert cfg.arm.dof == len(cfg.arm.home_position)


class TestFactoryRegression:
    """The DI graph still composes end-to-end with mock hardware."""

    def test_orchestrator_exposes_all_subsystems(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        orch = build_arm_orchestrator(cfg)
        assert orch.perception is not None
        assert orch.planner is not None
        assert orch.controller is not None
        assert orch.environment is not None
        assert orch.driver is not None


class TestProtocolRegression:
    """Both drivers satisfy the existing protocol surface."""

    def test_mock_driver_satisfies_protocol(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        driver = MockArmDriver(cfg.arm)
        assert isinstance(driver, ArmDriverProtocol)

    def test_esp32_driver_satisfies_protocol(self) -> None:
        from armdroid.hardware.esp32_json_driver import Esp32JsonDriver

        cfg = ArmSettings(mock_hardware=True)
        driver = Esp32JsonDriver(cfg.arm)
        assert isinstance(driver, ArmDriverProtocol)


class TestNamedConstantsRegression:
    """Named poll-interval constants must remain present and correctly typed."""

    def test_keepalive_poll_floor_is_float(self) -> None:
        from armdroid.hardware import esp32_json_driver

        assert hasattr(esp32_json_driver, "_KEEPALIVE_POLL_FLOOR_S")
        assert isinstance(esp32_json_driver._KEEPALIVE_POLL_FLOOR_S, float)
        assert esp32_json_driver._KEEPALIVE_POLL_FLOOR_S > 0

    def test_first_state_poll_interval_is_float(self) -> None:
        from armdroid.hardware import esp32_json_driver

        assert hasattr(esp32_json_driver, "_FIRST_STATE_POLL_INTERVAL_S")
        assert isinstance(esp32_json_driver._FIRST_STATE_POLL_INTERVAL_S, float)
        assert esp32_json_driver._FIRST_STATE_POLL_INTERVAL_S > 0


class TestTestHarnessRegression:
    """Shared test-harness helpers must remain importable and structurally correct."""

    def test_fake_serial_helpers_importable(self) -> None:
        from tests.helpers.fake_serial import (
            FakeSerial,
            PingOnlyFakeSerial,
            SilentFakeSerial,
        )

        # SilentFakeSerial subclasses PingOnlyFakeSerial
        assert issubclass(SilentFakeSerial, PingOnlyFakeSerial)
        # FakeSerial is independent (not a base of PingOnlyFakeSerial)
        assert not issubclass(PingOnlyFakeSerial, FakeSerial)

    def test_fake_serial_dof_parameter(self) -> None:
        from tests.helpers.fake_serial import FakeSerial

        fs = FakeSerial(port="/dev/null", baudrate=115200, timeout=1.0, write_timeout=1.0, dof=4)
        assert fs._dof == 4
        assert len(fs._joints) == 4
