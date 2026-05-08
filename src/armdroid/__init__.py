"""armdroid — robot arm manipulation platform.

SO-ARM100 hardware driver, MuJoCo simulation, SAC+HER policy training,
PDDL symbolic planning, and YOLO-based perception for Tower of Hanoi
and laundry sorting tasks.

The public API is defined in :mod:`armdroid.api` and re-exported here.
Internal layout (``armdroid.domain``, ``armdroid.hardware.*``,
``armdroid.orchestration.*``, …) is not covered by SemVer guarantees and
may change between minor versions.
"""

from __future__ import annotations

from armdroid.api import (
    ArmCommandRejected,
    ArmControllerProtocol,
    ArmDriverError,
    ArmDriverProtocol,
    ArmDroidError,
    ArmEnvironmentProtocol,
    ArmOrchestrator,
    ArmPerceptionProtocol,
    ArmPlannerProtocol,
    ArmSettings,
    ArmState,
    ConfigError,
    DetectedObject,
    PerceptionError,
    PlanningError,
    PlanStep,
    SymbolicState,
    __version__,
    build_arm_controller,
    build_arm_driver,
    build_arm_environment,
    build_arm_orchestrator,
    build_arm_perception,
    build_arm_planner,
)

__all__ = [
    "ArmCommandRejected",
    "ArmControllerProtocol",
    "ArmDriverError",
    "ArmDriverProtocol",
    "ArmDroidError",
    "ArmEnvironmentProtocol",
    "ArmOrchestrator",
    "ArmPerceptionProtocol",
    "ArmPlannerProtocol",
    "ArmSettings",
    "ArmState",
    "ConfigError",
    "DetectedObject",
    "PerceptionError",
    "PlanStep",
    "PlanningError",
    "SymbolicState",
    "__version__",
    "build_arm_controller",
    "build_arm_driver",
    "build_arm_environment",
    "build_arm_orchestrator",
    "build_arm_perception",
    "build_arm_planner",
]
