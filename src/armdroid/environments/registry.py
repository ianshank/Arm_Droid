"""Environment registry — Phase 2 plugin seam for Gymnasium task envs.

Built-in environments (``tower_of_hanoi``, ``laundry_sorting``) register
on import. Out-of-tree environments may be plugged in via the
``armdroid.environments`` entry-point group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from armdroid._registry import Registry
from armdroid.environments.laundry_sorting import LaundrySortingEnv
from armdroid.environments.tower_of_hanoi import TowerOfHanoiEnv

if TYPE_CHECKING:
    from armdroid.domain.protocols import ArmEnvironmentProtocol

_ENVIRONMENTS: Registry[type[ArmEnvironmentProtocol]] = Registry(
    kind="environment",
    entry_point_group="armdroid.environments",
)

_ENVIRONMENTS.register("tower_of_hanoi", TowerOfHanoiEnv)
_ENVIRONMENTS.register("laundry_sorting", LaundrySortingEnv)


def register_environment(name: str, factory: type[ArmEnvironmentProtocol]) -> None:
    """Register an environment class under ``name``."""
    _ENVIRONMENTS.register(name, factory)


def get_environment(name: str) -> type[ArmEnvironmentProtocol]:
    """Return the environment class registered under ``name``."""
    return _ENVIRONMENTS.get(name)


def available_environments() -> list[str]:
    """Return the sorted list of registered environment names."""
    return _ENVIRONMENTS.available()


def load_environment_plugins() -> int:
    """Discover and register out-of-tree environments via entry points."""
    return _ENVIRONMENTS.load_entry_points()


__all__ = [
    "available_environments",
    "get_environment",
    "load_environment_plugins",
    "register_environment",
]
