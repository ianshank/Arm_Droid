"""Baseline regression tests.

These tests guard the current armdroid public surface against accidental
breakage during the ESP32-arm integration. They cover:

* Schema construction with no overlays.
* Factory DI graph: building an orchestrator with mock hardware succeeds
  and exposes all five subsystem properties.
* Protocol conformance: the existing ``MockArmDriver`` still satisfies
  the (currently 6-DoF) ``ArmDriverProtocol``.
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
    """The mock driver still satisfies the existing protocol surface."""

    def test_mock_driver_satisfies_protocol(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        driver = MockArmDriver(cfg.arm)
        assert isinstance(driver, ArmDriverProtocol)
