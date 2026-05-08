"""Backwards-compatibility re-exports for the legacy ``armdroid.protocols`` path.

The canonical home of these symbols is :mod:`armdroid.domain` (split across
``state``, ``errors``, ``protocols`` for separation of concerns). This module
exists so existing imports continue to work for the v0.2.x line. A
``DeprecationWarning`` will be added in a later phase once consumers have
migrated; the import path itself is scheduled for removal in v0.4.0.

New code should import from :mod:`armdroid` (public façade) or, if a domain
type is genuinely internal-only, from :mod:`armdroid.domain` directly.
"""

from __future__ import annotations

from armdroid.domain.errors import ArmCommandRejected, ArmDriverError
from armdroid.domain.protocols import (
    ArmControllerProtocol,
    ArmDriverProtocol,
    ArmEnvironmentProtocol,
    ArmPerceptionProtocol,
    ArmPlannerProtocol,
)
from armdroid.domain.state import ArmState, DetectedObject, PlanStep, SymbolicState

__all__ = [
    "ArmCommandRejected",
    "ArmControllerProtocol",
    "ArmDriverError",
    "ArmDriverProtocol",
    "ArmEnvironmentProtocol",
    "ArmPerceptionProtocol",
    "ArmPlannerProtocol",
    "ArmState",
    "DetectedObject",
    "PlanStep",
    "SymbolicState",
]
