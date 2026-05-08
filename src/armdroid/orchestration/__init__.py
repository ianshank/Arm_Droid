"""Composition root: factory functions, orchestrator, lifecycle helpers.

This package wires concrete drivers, planners, perception backends, and
environments into a single :class:`armdroid.orchestration.orchestrator.ArmOrchestrator`
instance. It depends on every adapter package but is never imported by them
— keeping the dependency graph acyclic.

Public re-exports:
    :class:`ArmOrchestrator`
    :func:`build_arm_driver`, :func:`build_arm_planner`,
    :func:`build_arm_perception`, :func:`build_arm_controller`,
    :func:`build_arm_environment`, :func:`build_arm_orchestrator`

Phase 2 will introduce a ``lifecycle.py`` submodule (start / stop / shutdown
helpers extracted from the orchestrator) and registry-driven dispatch in
``factory.py``.
"""

from __future__ import annotations

from armdroid.orchestration.factory import (
    build_arm_controller,
    build_arm_driver,
    build_arm_environment,
    build_arm_orchestrator,
    build_arm_perception,
    build_arm_planner,
)
from armdroid.orchestration.orchestrator import ArmOrchestrator

__all__ = [
    "ArmOrchestrator",
    "build_arm_controller",
    "build_arm_driver",
    "build_arm_environment",
    "build_arm_orchestrator",
    "build_arm_perception",
    "build_arm_planner",
]
