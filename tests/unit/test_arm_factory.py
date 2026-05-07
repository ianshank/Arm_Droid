"""Tests for armdroid factory functions."""

from __future__ import annotations

from armdroid.config.schema import ArmSettings, ArmTaskConfig
from armdroid.factory import (
    build_arm_controller,
    build_arm_driver,
    build_arm_environment,
    build_arm_orchestrator,
    build_arm_perception,
    build_arm_planner,
)
from armdroid.hardware.mock_arm_driver import MockArmDriver
from armdroid.orchestrator import ArmOrchestrator
from armdroid.protocols import (
    ArmControllerProtocol,
    ArmDriverProtocol,
    ArmEnvironmentProtocol,
    ArmPerceptionProtocol,
    ArmPlannerProtocol,
)


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

    def test_controller_exposes_agent_and_primitives(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        controller = build_arm_controller(cfg)
        # public properties (used by orchestrator + factory wiring)
        assert controller.agent is not None  # type: ignore[attr-defined]
        assert controller.primitives is not None  # type: ignore[attr-defined]
        assert not controller.agent.is_built  # type: ignore[attr-defined]


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
        assert orch.driver is orch.controller.primitives.driver  # type: ignore[attr-defined]
