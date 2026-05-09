"""RL agent registry — Phase 2 plugin seam.

Built-in agents (``sac``, ``sac_her``) register on import. Out-of-tree
agents may be plugged in via the ``armdroid.rl_agents`` entry-point
group. PR-B registers ``rsl_rl_ppo`` here.

Closes G4: the registry generic is a callable factory that returns
:class:`armdroid.domain.protocols.ArmRLAgentProtocol`, so out-of-tree
agents must satisfy the contract on construction. Using ``Callable[...,
ArmRLAgentProtocol]`` rather than ``type[ArmRLAgentProtocol]`` accepts
both class objects (whose ``__init__`` becomes the factory) and plain
factory functions, AND it lets mypy verify ``agent_cls(cfg.arm_training)``
without a ``# type: ignore[call-arg]`` (Protocol classes don't constrain
``__init__``). PR-B's ``RslRlPpoAgent`` can register itself as a class
or as a wrapped factory without changing this surface.
"""

from __future__ import annotations

from collections.abc import Callable

from armdroid._registry import Registry
from armdroid.control.sac_agent import SACAgent
from armdroid.domain.protocols import ArmRLAgentProtocol

#: Type alias for the registry value — any callable returning an
#: :class:`ArmRLAgentProtocol` implementation. Class objects satisfy
#: this naturally (their ``__init__`` is the factory), as do plain
#: factory functions wrapping more complex constructors.
ArmRLAgentFactory = Callable[..., ArmRLAgentProtocol]


_AGENTS: Registry[ArmRLAgentFactory] = Registry(
    kind="rl_agent",
    entry_point_group="armdroid.rl_agents",
)

_AGENTS.register("sac", SACAgent)
# ``sac_her`` alias matches the default value of
# ``ArmTrainingConfig.algorithm`` so build_arm_controller can resolve the
# default config directly through the registry without a hardcoded
# string-compare. Both names point at the same SACAgent (SAC+HER under
# the hood — the agent class wires HerReplayBuffer in build()).
# Mirrored in pyproject.toml's [project.entry-points."armdroid.rl_agents"].
_AGENTS.register("sac_her", SACAgent)


def register_rl_agent(name: str, factory: ArmRLAgentFactory) -> None:
    """Register an RL agent factory under ``name``.

    ``factory`` may be a class object (its ``__init__`` is the factory)
    or any callable returning an :class:`ArmRLAgentProtocol`.
    """
    _AGENTS.register(name, factory)


def get_rl_agent(name: str) -> ArmRLAgentFactory:
    """Return the RL agent factory registered under ``name``."""
    return _AGENTS.get(name)


def available_rl_agents() -> list[str]:
    """Return the sorted list of registered RL agent names."""
    return _AGENTS.available()


def load_rl_agent_plugins() -> int:
    """Discover and register out-of-tree RL agents via entry points."""
    return _AGENTS.load_entry_points()


__all__ = [
    "ArmRLAgentFactory",
    "available_rl_agents",
    "get_rl_agent",
    "load_rl_agent_plugins",
    "register_rl_agent",
]
