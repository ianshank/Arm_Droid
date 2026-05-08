"""RL agent registry — Phase 2 plugin seam.

Built-in agents (``sac``) register on import. Out-of-tree agents may be
plugged in via the ``armdroid.rl_agents`` entry-point group.
"""

from __future__ import annotations

from typing import Any

from armdroid._registry import Registry
from armdroid.control.sac_agent import SACAgent

# RL agent classes do not yet share a single Protocol (SB3 typing
# constraints). Stored as ``type[Any]`` until Phase 4 introduces
# ``ArmRLAgentProtocol`` in ``armdroid.domain.protocols``.
_AGENTS: Registry[type[Any]] = Registry(
    kind="rl_agent",
    entry_point_group="armdroid.rl_agents",
)

_AGENTS.register("sac", SACAgent)


def register_rl_agent(name: str, factory: type[Any]) -> None:
    """Register an RL agent class under ``name``."""
    _AGENTS.register(name, factory)


def get_rl_agent(name: str) -> type[Any]:
    """Return the RL agent class registered under ``name``."""
    return _AGENTS.get(name)


def available_rl_agents() -> list[str]:
    """Return the sorted list of registered RL agent names."""
    return _AGENTS.available()


def load_rl_agent_plugins() -> int:
    """Discover and register out-of-tree RL agents via entry points."""
    return _AGENTS.load_entry_points()


__all__ = [
    "available_rl_agents",
    "get_rl_agent",
    "load_rl_agent_plugins",
    "register_rl_agent",
]
