"""Perception backend registry — Phase 2 plugin seam.

Currently the perception pipeline is a single facade
(:class:`armdroid.perception.facade.ArmPerception`) composed of
sub-components (depth processor, YOLO detector, pose estimator, state
extractor). Phase 2b will introduce ``kind`` discriminators that select
camera and detector backends individually; for now the registry exposes
the facade as the ``default`` entry so plugin authors can already wire
through entry points.

Out-of-tree backends may be plugged in via the
``armdroid.perception_backends`` entry-point group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from armdroid._registry import Registry
from armdroid.perception.facade import ArmPerception

if TYPE_CHECKING:
    from armdroid.domain.protocols import ArmPerceptionProtocol

_PERCEPTION: Registry[type[ArmPerceptionProtocol]] = Registry(
    kind="perception_backend",
    entry_point_group="armdroid.perception_backends",
)

_PERCEPTION.register("default", ArmPerception)


def register_perception_backend(name: str, factory: type[ArmPerceptionProtocol]) -> None:
    """Register a perception backend class under ``name``."""
    _PERCEPTION.register(name, factory)


def get_perception_backend(name: str) -> type[ArmPerceptionProtocol]:
    """Return the perception backend class registered under ``name``."""
    return _PERCEPTION.get(name)


def available_perception_backends() -> list[str]:
    """Return the sorted list of registered perception backend names."""
    return _PERCEPTION.available()


def load_perception_plugins() -> int:
    """Discover and register out-of-tree perception backends via entry points."""
    return _PERCEPTION.load_entry_points()


__all__ = [
    "available_perception_backends",
    "get_perception_backend",
    "load_perception_plugins",
    "register_perception_backend",
]
