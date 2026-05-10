"""RL agent registry — Phase 2 plugin seam.

Built-in agents (``sac``, ``sac_her``) register on import. Out-of-tree
agents may be plugged in via the ``armdroid.rl_agents`` entry-point
group. PR-B registers ``rsl_rl_ppo`` here.

The registry generic is a callable factory that returns
:class:`armdroid.domain.protocols.ArmRLAgentProtocol`, so out-of-tree
agents must satisfy the contract on construction (verified by
``tests/regression/test_baseline.py::test_sac_agent_satisfies_protocol``
+ ``tests/unit/test_protocols.py``). Using ``Callable[...,
ArmRLAgentProtocol]`` rather than ``type[ArmRLAgentProtocol]`` accepts
both class objects (whose ``__init__`` becomes the factory) and plain
factory functions, AND it lets mypy verify ``agent_cls(cfg.arm_training)``
without a ``# type: ignore[call-arg]`` (Protocol classes don't constrain
``__init__``). PR-B's ``RslRlPpoAgent`` registers as a wrapped factory
because its constructor takes both ``training_cfg`` and ``ppo_cfg`` —
see ``_rsl_rl_ppo_factory`` below.

Both built-in agent classes (``SACAgent``, ``RslRlPpoAgent``) declare
``build(env: ArmEnvironmentProtocol) -> None`` (TD-4): runtime
isinstance checks pass and mypy strict verifies signature variance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from armdroid._registry import Registry
from armdroid.control.sac_agent import SACAgent
from armdroid.domain.protocols import ArmRLAgentProtocol

if TYPE_CHECKING:
    from armdroid.config.schema.training import ArmTrainingConfig

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


def _rsl_rl_ppo_factory(training_cfg: ArmTrainingConfig) -> ArmRLAgentProtocol:
    """Lazy factory placeholder for RslRlPpoAgent (PR-B B.13).

    The actual RslRlPpoAgent constructor takes BOTH ``training_cfg``
    AND ``ppo_cfg``; the registry's
    ``Callable[..., ArmRLAgentProtocol]`` generic cannot accommodate
    dual-config without re-reading ArmSettings (which would drop YAML
    overlays — peer-review C-1). This factory is the compatibility
    layer for ``test_entry_point_mirror`` and
    ``available_rl_agents()`` introspection; the orchestrator's
    ``build_arm_controller`` branches on
    ``algorithm == "rsl_rl_ppo"`` and instantiates ``RslRlPpoAgent``
    directly with both configs.

    Calling this factory directly (outside build_arm_controller) raises
    NotImplementedError pointing at the correct construction path.
    """
    msg = (
        "RslRlPpoAgent requires both training_cfg AND ppo_cfg. "
        "Use armdroid.orchestration.factory.build_arm_controller(cfg) "
        "which threads both configs through automatically. Direct "
        "registry-dispatched construction is not supported for "
        "rsl_rl_ppo (peer-review C-1)."
    )
    raise NotImplementedError(msg)


_AGENTS.register("rsl_rl_ppo", _rsl_rl_ppo_factory)


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
