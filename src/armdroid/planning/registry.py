"""Planner registry — Phase 2 plugin seam for symbolic / LLM planners.

Built-in planners (``pyperplan``) register on import. Out-of-tree planners
may be plugged in via the ``armdroid.planners`` entry-point group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from armdroid._registry import Registry
from armdroid.planning.symbolic_planner import SymbolicPlanner

if TYPE_CHECKING:
    from armdroid.domain.protocols import ArmPlannerProtocol

_PLANNERS: Registry[type[ArmPlannerProtocol]] = Registry(
    kind="planner",
    entry_point_group="armdroid.planners",
)

# ``SymbolicPlanner`` currently dispatches between pyperplan and an LLM
# replanner internally based on cfg.planner_backend. In Phase 2b we will
# split those into separate registry entries (``llm:anthropic``, etc.).
_PLANNERS.register("pyperplan", SymbolicPlanner)


def register_planner(name: str, factory: type[ArmPlannerProtocol]) -> None:
    """Register a planner class under ``name``."""
    _PLANNERS.register(name, factory)


def get_planner(name: str) -> type[ArmPlannerProtocol]:
    """Return the planner class registered under ``name``."""
    return _PLANNERS.get(name)


def available_planners() -> list[str]:
    """Return the sorted list of registered planner names."""
    return _PLANNERS.available()


def load_planner_plugins() -> int:
    """Discover and register out-of-tree planners via entry points."""
    return _PLANNERS.load_entry_points()


__all__ = [
    "available_planners",
    "get_planner",
    "load_planner_plugins",
    "register_planner",
]
