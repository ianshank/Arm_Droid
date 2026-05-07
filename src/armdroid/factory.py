"""Factory functions for armdroid components.

Single wiring point for all armdroid subsystems. Each ``build_*()``
function returns the relevant protocol type so callers depend only on
:mod:`armdroid.protocols`.

Adapted from ``mousedroid/factory.py:2022-2182`` with:
- The ``hailo_runtime`` parameter dropped from ``build_arm_perception``
  (Hailo was a Jetson-only accelerator).
- All previously-lazy concrete imports hoisted to module top level
  (the relevant deps are now in ``[project.dependencies]`` not optional).
- ``cfg: Settings`` → ``cfg: ArmSettings``. Sub-configs default to
  populated instances so the rover's ``if cfg.arm is None`` guards
  are no longer needed.
- ``build_arm_orchestrator`` added as the top-level composition point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from armdroid.control.controller import ArmController
from armdroid.control.primitives import ActionPrimitives
from armdroid.control.sac_agent import SACAgent
from armdroid.environments.laundry_sorting import LaundrySortingEnv
from armdroid.environments.tower_of_hanoi import TowerOfHanoiEnv
from armdroid.hardware.mock_arm_driver import MockArmDriver
from armdroid.hardware.so_arm100_driver import SoArm100Driver
from armdroid.logging.setup import get_logger
from armdroid.orchestrator import ArmOrchestrator
from armdroid.perception.facade import ArmPerception
from armdroid.planning.symbolic_planner import SymbolicPlanner
from armdroid.protocols import (
    ArmControllerProtocol,
    ArmDriverProtocol,
    ArmEnvironmentProtocol,
    ArmPerceptionProtocol,
    ArmPlannerProtocol,
)

if TYPE_CHECKING:
    from armdroid.config.schema import ArmSettings

_log = get_logger(__name__)


def build_arm_driver(cfg: ArmSettings) -> ArmDriverProtocol:
    """Build the robot-arm hardware driver.

    Args:
        cfg: Root settings.

    Returns:
        Driver conforming to :class:`ArmDriverProtocol`.
    """
    if cfg.mock_hardware:
        _log.info("arm_driver_mock_built")
        return MockArmDriver(cfg.arm)

    _log.info("arm_driver_real_built", port=cfg.arm.serial_port)
    return SoArm100Driver(cfg.arm)


def build_arm_planner(cfg: ArmSettings) -> ArmPlannerProtocol:
    """Build the symbolic planner for arm manipulation tasks.

    Args:
        cfg: Root settings.

    Returns:
        Planner conforming to :class:`ArmPlannerProtocol`.
    """
    _log.info("arm_planner_built", backend=cfg.arm_planning.planner_backend)
    return SymbolicPlanner(cfg.arm_planning, cfg.arm_task)


def build_arm_environment(cfg: ArmSettings) -> ArmEnvironmentProtocol:
    """Build the Gymnasium environment for arm training.

    Args:
        cfg: Root settings.

    Returns:
        Environment conforming to :class:`ArmEnvironmentProtocol`.
    """
    dof = cfg.arm.dof

    if cfg.arm_task.task_type == "laundry_sorting":
        _log.info("arm_env_laundry_built")
        return LaundrySortingEnv(cfg.arm_task, cfg.arm_training, dof=dof)

    _log.info("arm_env_hanoi_built", num_disks=cfg.arm_task.num_disks)
    return TowerOfHanoiEnv(cfg.arm_task, cfg.arm_training, dof=dof)


def build_arm_controller(cfg: ArmSettings) -> ArmControllerProtocol:
    """Build the RL controller (SACAgent + ActionPrimitives).

    The returned controller's underlying :class:`SACAgent` is **not yet
    bound to an environment** — the orchestrator calls
    ``controller.agent.build(env)`` after the env is constructed.

    Args:
        cfg: Root settings.

    Returns:
        Controller conforming to :class:`ArmControllerProtocol`.
    """
    driver = build_arm_driver(cfg)
    agent = SACAgent(cfg.arm_training)
    primitives = ActionPrimitives(cfg.arm, driver)
    _log.info("arm_controller_built", algorithm=cfg.arm_training.algorithm)
    return ArmController(agent, primitives)


def build_arm_perception(cfg: ArmSettings) -> ArmPerceptionProtocol:
    """Build the perception pipeline (depth + YOLO + pose + symbolic state).

    Uses the ultralytics-on-CUDA YOLO detector path. The Hailo-accelerator
    branch from rover has been removed (Jetson-only hardware).

    Args:
        cfg: Root settings.

    Returns:
        Perception facade conforming to :class:`ArmPerceptionProtocol`.
    """
    intrinsics = np.eye(3, dtype=np.float64)
    intrinsics[0, 0] = cfg.arm_perception.default_focal_length
    intrinsics[1, 1] = cfg.arm_perception.default_focal_length
    intrinsics[0, 2] = cfg.arm_perception.default_principal_x
    intrinsics[1, 2] = cfg.arm_perception.default_principal_y

    _log.info("arm_perception_built", camera=cfg.arm_perception.depth_camera_type)
    return ArmPerception(cfg.arm_perception, cfg.arm_task, intrinsics)


def build_arm_orchestrator(cfg: ArmSettings) -> ArmOrchestrator:
    """Compose the full armdroid orchestrator from sub-component factories.

    Args:
        cfg: Root settings.

    Returns:
        :class:`ArmOrchestrator` with perception, planner, controller,
        environment, and driver wired up. The SAC agent inside the
        controller is not yet built — the orchestrator wires the env
        in :meth:`ArmOrchestrator.train` lazily.
    """
    perception = build_arm_perception(cfg)
    planner = build_arm_planner(cfg)
    controller = build_arm_controller(cfg)
    environment = build_arm_environment(cfg)
    # Reuse the controller's already-built driver instead of constructing
    # a second one. Keeps the SO-ARM100 serial port lock single-owner.
    driver = controller.primitives.driver  # type: ignore[attr-defined]
    _log.info("arm_orchestrator_built")
    return ArmOrchestrator(
        perception=perception,
        planner=planner,
        controller=controller,
        environment=environment,
        driver=driver,
    )
