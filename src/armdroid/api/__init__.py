"""Stable public API for the armdroid platform.

This is the only path external consumers should import from. Items here are
covered by SemVer guarantees on the ``armdroid`` distribution; everything
else (``armdroid.domain``, ``armdroid.hardware.*``, ``armdroid.orchestration.*``,
…) is subject to change between minor versions.

Re-exports:
    :class:`ArmOrchestrator`
    :class:`ArmSettings`
    :func:`build_arm_orchestrator`, :func:`build_arm_driver`,
    :func:`build_arm_planner`, :func:`build_arm_perception`,
    :func:`build_arm_controller`, :func:`build_arm_environment`
    Protocol contracts: :class:`ArmDriverProtocol`,
    :class:`ArmPerceptionProtocol`, :class:`ArmPlannerProtocol`,
    :class:`ArmControllerProtocol`, :class:`ArmEnvironmentProtocol`,
    :class:`VisionLanguageAgentProtocol` (Phase A scaffolding; Phase D
    implementation), :class:`HighLevelPlannerProtocol`,
    :class:`SafetyGuardProtocol`, :class:`InteractionSessionProtocol`
    Value types: :class:`ArmAction`, :class:`ArmState`,
    :class:`DetectedObject`, :class:`InteractionEvent`,
    :class:`PlanStep`, :class:`SceneInsight`, :class:`SymbolicState`,
    :class:`Verdict`
    Errors: :class:`ArmDroidError` (root), :class:`ConfigError`,
    :class:`ArmDriverError`, :class:`ArmCommandRejected`,
    :class:`PerceptionError`, :class:`PlanningError`
    :data:`__version__`
"""

from __future__ import annotations

from armdroid.api.version import __version__
from armdroid.config.schema import ArmSettings
from armdroid.domain.errors import (
    ArmCommandRejected,
    ArmDriverError,
    ArmDroidError,
    ConfigError,
    PerceptionError,
    PlanningError,
)
from armdroid.domain.protocols import (
    ArmControllerProtocol,
    ArmDriverProtocol,
    ArmEnvironmentProtocol,
    ArmPerceptionProtocol,
    ArmPlannerProtocol,
    HighLevelPlannerProtocol,
    InteractionSessionProtocol,
    SafetyGuardProtocol,
    VisionLanguageAgentProtocol,
)
from armdroid.domain.state import (
    ArmAction,
    ArmState,
    DetectedObject,
    InteractionEvent,
    PlanStep,
    SceneInsight,
    SymbolicState,
    Verdict,
)
from armdroid.orchestration import (
    ArmOrchestrator,
    build_arm_controller,
    build_arm_driver,
    build_arm_environment,
    build_arm_orchestrator,
    build_arm_perception,
    build_arm_planner,
)

__all__ = [
    "ArmAction",
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
    "HighLevelPlannerProtocol",
    "InteractionEvent",
    "InteractionSessionProtocol",
    "PerceptionError",
    "PlanStep",
    "PlanningError",
    "SafetyGuardProtocol",
    "SceneInsight",
    "SymbolicState",
    "Verdict",
    "VisionLanguageAgentProtocol",
    "__version__",
    "build_arm_controller",
    "build_arm_driver",
    "build_arm_environment",
    "build_arm_orchestrator",
    "build_arm_perception",
    "build_arm_planner",
]
