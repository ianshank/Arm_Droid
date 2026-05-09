"""RL agent registry — Phase 2 plugin seam.

Built-in agents (``sac``, ``sac_her``) register on import. Out-of-tree
agents may be plugged in via the ``armdroid.rl_agents`` entry-point
group. PR-B registers ``rsl_rl_ppo`` here.

Closes G4: the registry generic is tightened to
:class:`armdroid.domain.protocols.ArmRLAgentProtocol` so out-of-tree
agents are forced to satisfy the contract instead of being typed as
``Any``.
"""

from __future__ import annotations

from armdroid._registry import Registry
from armdroid.control.sac_agent import SACAgent
from armdroid.domain.protocols import ArmRLAgentProtocol

_AGENTS: Registry[type[ArmRLAgentProtocol]] = Registry(
    kind="rl_agent",
    entry_point_group="armdroid.rl_agents",
)

_AGENTS.register("sac", SACAgent)
# ``sac_her`` alias matches the default value of
# ``ArmTrainingConfig.algorithm`` so build_arm_controller can resolve the
# default config directly through the registry without a hardcoded
# string-compare. Both names point at the same SACAgent (SAC+HER under
# the hood — the agent class wires HerReplayBuffer in build()).
_AGENTS.register("sac_her", SACAgent)


def register_rl_agent(name: str, factory: type[ArmRLAgentProtocol]) -> None:
    """Register an RL agent class under ``name``."""
    _AGENTS.register(name, factory)


def get_rl_agent(name: str) -> type[ArmRLAgentProtocol]:
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
