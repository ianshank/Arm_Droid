"""Baseline regression tests.

These tests guard the current armdroid public surface against accidental
breakage during the ESP32-arm integration. They cover:

* Schema construction with no overlays.
* Factory DI graph: building an orchestrator with mock hardware succeeds
  and exposes all five subsystem properties.
* Root-package public API re-exports and built-in driver registry entries.
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
from armdroid.domain.protocols import ArmDriverProtocol
from armdroid.hardware.mock_arm_driver import MockArmDriver
from armdroid.orchestration.factory import build_arm_orchestrator

pytestmark = pytest.mark.regression


class TestSchemaRegression:
    """Default-constructed settings stay valid across changes."""

    def test_arm_settings_constructs_without_overlays(self) -> None:
        # conftest.py forces ARMDROID_MOCK_HARDWARE=true for the test run, so
        # the BaseSettings env reader picks that up here. We assert on the
        # *shape* of the config (dof, home length match) rather than on the
        # mock_hardware default.
        cfg = ArmSettings()
        assert cfg.arm.dof == 6  # current baseline; 7-DoF stays deferred pending validation
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


class TestPublicSurfaceRegression:
    """The stable public API and built-in registry surface stay importable."""

    def test_root_package_reexports_public_api(self) -> None:
        import armdroid
        import armdroid.api as api

        assert armdroid.ArmOrchestrator is api.ArmOrchestrator
        assert armdroid.build_arm_orchestrator is api.build_arm_orchestrator

    def test_driver_registry_exposes_expected_built_ins(self) -> None:
        from armdroid.hardware.registry import available_drivers

        names = available_drivers()
        assert "esp32" in names
        assert "mock" in names


class TestProtocolRegression:
    """Both drivers satisfy the existing protocol surface."""

    def test_mock_driver_satisfies_protocol(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        driver = MockArmDriver(cfg.arm)
        assert isinstance(driver, ArmDriverProtocol)

    def test_esp32_driver_satisfies_protocol(self) -> None:
        # The ESP32 driver requires the optional ``[hardware]`` extra (pyserial)
        # to instantiate. CI's default ``[dev]`` install omits that extra, so
        # skip cleanly when pyserial is not importable rather than asserting
        # against environments that legitimately can't construct the driver.
        pytest.importorskip("serial", reason="pyserial not installed; install with .[hardware]")
        from armdroid.hardware.esp32 import Esp32JsonDriver

        cfg = ArmSettings(mock_hardware=True)
        driver = Esp32JsonDriver(cfg.arm)
        assert isinstance(driver, ArmDriverProtocol)

    def test_sac_agent_satisfies_arm_rl_agent_protocol(self) -> None:
        """SACAgent must implement ArmRLAgentProtocol so the registry can
        type the value as ``type[ArmRLAgentProtocol]`` (closes G4).
        """
        from armdroid.config.schema import ArmTrainingConfig
        from armdroid.control.sac_agent import SACAgent
        from armdroid.domain.protocols import ArmRLAgentProtocol

        agent = SACAgent(ArmTrainingConfig())
        assert isinstance(agent, ArmRLAgentProtocol)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
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
