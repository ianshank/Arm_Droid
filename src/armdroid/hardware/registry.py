"""Driver registry — Phase 2 plugin seam for arm hardware backends.

Built-in drivers (``mock``, ``esp32``) are registered on import. Out-of-tree
drivers can be plugged in via the ``armdroid.drivers`` entry-point group;
call :func:`load_driver_plugins` once at process start to discover them.

The registry stores driver *classes* (not instances). The factory in
:mod:`armdroid.orchestration.factory` will, in Phase 2b, switch to
``get_driver(cfg.driver_kind)(cfg.arm)`` instead of the current ``if
cfg.mock_hardware`` branch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from armdroid._registry import Registry
from armdroid.hardware.esp32 import Esp32JsonDriver
from armdroid.hardware.mock_arm_driver import MockArmDriver

if TYPE_CHECKING:
    from armdroid.domain.protocols import ArmDriverProtocol

_DRIVERS: Registry[type[ArmDriverProtocol]] = Registry(
    kind="driver",
    entry_point_group="armdroid.drivers",
)

_DRIVERS.register("mock", MockArmDriver)
_DRIVERS.register("esp32", Esp32JsonDriver)


def register_driver(name: str, factory: type[ArmDriverProtocol]) -> None:
    """Register a driver class under ``name``."""
    _DRIVERS.register(name, factory)


def get_driver(name: str) -> type[ArmDriverProtocol]:
    """Return the driver class registered under ``name``."""
    return _DRIVERS.get(name)


def available_drivers() -> list[str]:
    """Return the sorted list of registered driver names."""
    return _DRIVERS.available()


def load_driver_plugins() -> int:
    """Discover and register out-of-tree drivers via entry points."""
    return _DRIVERS.load_entry_points()


__all__ = [
    "available_drivers",
    "get_driver",
    "load_driver_plugins",
    "register_driver",
]
