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
    :class:`ArmControllerProtocol`, :class:`ArmEnvironmentProtocol`
    Value types: :class:`ArmState`, :class:`DetectedObject`,
    :class:`SymbolicState`, :class:`PlanStep`
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
)
from armdroid.domain.state import ArmState, DetectedObject, PlanStep, SymbolicState
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
