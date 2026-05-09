"""Tests for armdroid factory functions."""

from __future__ import annotations

import pytest

from armdroid._registry import RegistryError
from armdroid.config.schema import ArmSettings, ArmTaskConfig
from armdroid.control.controller import ArmController
from armdroid.domain.protocols import (
    ArmControllerProtocol,
    ArmDriverProtocol,
    ArmEnvironmentProtocol,
    ArmPerceptionProtocol,
    ArmPlannerProtocol,
)
from armdroid.hardware.mock_arm_driver import MockArmDriver
from armdroid.orchestration.factory import (
    build_arm_controller,
    build_arm_driver,
    build_arm_environment,
    build_arm_orchestrator,
    build_arm_perception,
    build_arm_planner,
)
from armdroid.orchestration.orchestrator import ArmOrchestrator


class TestBuildArmDriver:
    """build_arm_driver returns a mock driver when mock_hardware=True."""

    def test_mock_returns_mock_driver(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        driver = build_arm_driver(cfg)
        assert isinstance(driver, MockArmDriver)
        assert isinstance(driver, ArmDriverProtocol)


class TestBuildArmPlanner:
    """build_arm_planner constructs the symbolic planner."""

    def test_builds_symbolic_planner(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        planner = build_arm_planner(cfg)
        assert isinstance(planner, ArmPlannerProtocol)


class TestBuildArmEnvironment:
    """build_arm_environment dispatches on task_type."""

    def test_builds_hanoi_env(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        env = build_arm_environment(cfg)
        assert isinstance(env, ArmEnvironmentProtocol)

    def test_builds_laundry_env(self) -> None:
        cfg = ArmSettings(
            mock_hardware=True,
            arm_task=ArmTaskConfig(task_type="laundry_sorting"),
        )
        env = build_arm_environment(cfg)
        assert isinstance(env, ArmEnvironmentProtocol)


class TestBuildArmController:
    """build_arm_controller wires SACAgent + ActionPrimitives."""

    def test_builds_controller(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        controller = build_arm_controller(cfg)
        assert isinstance(controller, ArmControllerProtocol)
        assert isinstance(controller, ArmController)

    def test_controller_exposes_agent_and_primitives(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        controller = build_arm_controller(cfg)
        assert isinstance(controller, ArmController)
        assert controller.agent is not None
        assert controller.primitives is not None
        assert not controller.agent.is_built

    def test_controller_accepts_explicit_driver(self) -> None:
        """Passing an existing driver avoids opening a second serial connection."""
        cfg = ArmSettings(mock_hardware=True)
        shared_driver = build_arm_driver(cfg)
        controller = build_arm_controller(cfg, driver=shared_driver)
        assert isinstance(controller, ArmController)
        assert controller.primitives.driver is shared_driver

    def test_default_algorithm_dispatches_via_registry(self) -> None:
        """``cfg.arm_training.algorithm`` (default 'sac_her') resolves through
        the registry, not via a hardcoded import.

        Closes preparation seam for PR-B's rsl_rl_ppo agent registration.
        """
        from armdroid.control.sac_agent import SACAgent

        cfg = ArmSettings(arm_driver_kind="mock")
        # Default algorithm is sac_her; both sac and sac_her registered to SACAgent.
        assert cfg.arm_training.algorithm == "sac_her"
        controller = build_arm_controller(cfg)
        assert isinstance(controller, ArmController)
        assert isinstance(controller.agent, SACAgent)

    def test_explicit_sac_algorithm_dispatches_via_registry(self) -> None:
        """Setting algorithm='sac' uses the SACAgent registered under 'sac'."""
        from armdroid.config.schema import ArmTrainingConfig
        from armdroid.control.sac_agent import SACAgent

        # Override training to algorithm='sac' (the explicit, non-HER label)
        cfg = ArmSettings(
            arm_driver_kind="mock",
            arm_training=ArmTrainingConfig(algorithm="sac"),
        )
        controller = build_arm_controller(cfg)
        assert isinstance(controller.agent, SACAgent)


class TestBuildArmPerception:
    """build_arm_perception constructs the perception facade."""

    def test_builds_perception(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        perception = build_arm_perception(cfg)
        assert isinstance(perception, ArmPerceptionProtocol)


class TestBuildArmOrchestrator:
    """build_arm_orchestrator composes the full DI graph."""

    def test_builds_full_orchestrator(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        orch = build_arm_orchestrator(cfg)
        assert isinstance(orch, ArmOrchestrator)
        assert isinstance(orch.perception, ArmPerceptionProtocol)
        assert isinstance(orch.planner, ArmPlannerProtocol)
        assert isinstance(orch.controller, ArmControllerProtocol)
        assert isinstance(orch.environment, ArmEnvironmentProtocol)
        assert isinstance(orch.driver, ArmDriverProtocol)

    def test_driver_is_shared_with_controller(self) -> None:
        """Single driver instance — same object held by controller's primitives."""
        cfg = ArmSettings(mock_hardware=True)
        orch = build_arm_orchestrator(cfg)
        ctrl = orch.controller
        assert isinstance(ctrl, ArmController)
        assert orch.driver is ctrl.primitives.driver


class TestRegistryDispatch:
    """Phase 2b: factory uses registry, unknown kinds raise RegistryError."""

    def test_unknown_driver_kind_raises(self) -> None:
        from armdroid.hardware.registry import get_driver

        with pytest.raises(RegistryError, match="unknown driver"):
            get_driver("nonexistent_driver")

    def test_unknown_environment_kind_raises(self) -> None:
        from armdroid.environments.registry import get_environment

        with pytest.raises(RegistryError, match="unknown environment"):
            get_environment("pick_place")
