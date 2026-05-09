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
from armdroid.control.registry import get_rl_agent
from armdroid.domain.protocols import (
    ArmControllerProtocol,
    ArmDriverProtocol,
    ArmEnvironmentProtocol,
    ArmPerceptionProtocol,
    ArmPlannerProtocol,
)
from armdroid.environments.registry import get_environment
from armdroid.hardware.registry import get_driver
from armdroid.logging.setup import get_logger
from armdroid.orchestration._driver_kind import resolve_driver_kind
from armdroid.orchestration.orchestrator import ArmOrchestrator
from armdroid.perception.facade import ArmPerception
from armdroid.planning.symbolic_planner import SymbolicPlanner

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
    kind = resolve_driver_kind(cfg)
    _log.info("arm_driver_built", kind=kind)
    return get_driver(kind)(cfg.arm)


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
    task_type = cfg.arm_task.task_type
    _log.info("arm_env_built", task_type=task_type)
    return get_environment(task_type)(cfg.arm_task, cfg.arm_training, dof=dof)


def build_arm_controller(
    cfg: ArmSettings,
    driver: ArmDriverProtocol | None = None,
) -> ArmControllerProtocol:
    """Build the RL controller (SACAgent + ActionPrimitives).

    The returned controller's underlying :class:`SACAgent` is **not yet
    bound to an environment** — call :meth:`build_for_env` (via the
    orchestrator's :meth:`train`) after the environment is constructed.

    Args:
        cfg: Root settings.
        driver: Pre-built driver to reuse. If ``None``, a new driver is
            constructed from ``cfg``. Pass an existing driver to avoid
            opening a second serial connection to the SO-ARM100.

    Returns:
        Controller conforming to :class:`ArmControllerProtocol`.
    """
    resolved_driver = driver if driver is not None else build_arm_driver(cfg)
    # Dispatch via the registry so PR-B can register rsl_rl_ppo as a
    # sibling without touching this code. SACAgent is registered under
    # both ``"sac"`` and ``"sac_her"`` (the default). The registry's
    # ``Callable[..., ArmRLAgentProtocol]`` factory type lets mypy verify
    # this call without ``# type: ignore[call-arg]`` — Protocol classes
    # don't constrain ``__init__`` directly, but factory-typed callables
    # accept any args. (PR #10 review: peer-review C-Copilot.)
    agent_factory = get_rl_agent(cfg.arm_training.algorithm)
    agent = agent_factory(cfg.arm_training)
    primitives = ActionPrimitives(cfg.arm, resolved_driver)
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

    The driver is constructed once here and shared with the controller so
    that only a single serial connection is opened to the SO-ARM100.

    Args:
        cfg: Root settings.

    Returns:
        :class:`ArmOrchestrator` with perception, planner, controller,
        environment, and driver wired up. The SAC agent inside the
        controller is not yet built — call :meth:`ArmOrchestrator.train`
        which delegates to :meth:`ArmControllerProtocol.build_for_env`
        lazily.
    """
    driver = build_arm_driver(cfg)
    perception = build_arm_perception(cfg)
    planner = build_arm_planner(cfg)
    controller = build_arm_controller(cfg, driver=driver)
    environment = build_arm_environment(cfg)
    _log.info("arm_orchestrator_built")
    return ArmOrchestrator(
        perception=perception,
        planner=planner,
        controller=controller,
        environment=environment,
        driver=driver,
    )
