"""Pure, framework-free domain layer: types, errors, and protocol contracts.

Modules in this package have no I/O, no third-party framework imports beyond
``numpy`` (used in array-shaped value types), and no dependency on other
``armdroid`` subsystems. They define the vocabulary every other layer speaks.

Submodules:
    :mod:`armdroid.domain.state`      — value objects (ArmState, DetectedObject, …)
    :mod:`armdroid.domain.errors`     — exception hierarchy
    :mod:`armdroid.domain.protocols`  — runtime-checkable Protocol interfaces
    :mod:`armdroid.domain.units`      — typed unit wrappers (Phase 4 expansion)

Stable public re-exports live at :mod:`armdroid` (the API façade). The
top-level :mod:`armdroid.protocols` shim re-exports from this package for
backwards compatibility and will continue to do so for the v0.2.x line.
"""

from __future__ import annotations

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

__all__ = [
    "ArmAction",
    "ArmCommandRejected",
    "ArmControllerProtocol",
    "ArmDriverError",
    "ArmDriverProtocol",
    "ArmDroidError",
    "ArmEnvironmentProtocol",
    "ArmPerceptionProtocol",
    "ArmPlannerProtocol",
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
]
