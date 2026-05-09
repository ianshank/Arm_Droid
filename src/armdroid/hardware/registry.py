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

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

from armdroid._registry import Registry
from armdroid.hardware.esp32 import Esp32JsonDriver
from armdroid.hardware.mock_arm_driver import MockArmDriver

if TYPE_CHECKING:
    from armdroid.config.schema import ArmConfig
    from armdroid.domain.protocols import ArmDriverProtocol

#: A driver factory: a callable that accepts an :class:`ArmConfig` and
#: returns a fully-constructed :class:`ArmDriverProtocol` instance.
DriverFactory: TypeAlias = "Callable[[ArmConfig], ArmDriverProtocol]"

_DRIVERS: Registry[DriverFactory] = Registry(
    kind="driver",
    entry_point_group="armdroid.drivers",
)

_DRIVERS.register("mock", MockArmDriver)
_DRIVERS.register("esp32", Esp32JsonDriver)


def _isaac_sim_factory(arm_cfg: ArmConfig) -> ArmDriverProtocol:
    """Lazy factory for IsaacSimDriver.

    Wrapping the import inside this function (rather than at module top)
    keeps ``armdroid.hardware.registry`` importable on default installs
    without the [isaac] extra. The actual import only happens when
    ``get_driver("isaac_sim")(cfg.arm)`` is called.

    Raises:
        ArmDriverError: When the [isaac] extra is not installed
            (isaaclab cannot be imported), with a hint pointing at the
            install command + NVIDIA pip index URL.
    """
    try:
        # Imports the package's lazy ``IsaacSimDriver`` via ``__getattr__``;
        # the cast tells mypy the resolved attribute is the driver class.
        from armdroid.hardware.isaac_sim.driver import IsaacSimDriver
    except ImportError as exc:
        from armdroid.domain.errors import ArmDriverError

        msg = (
            f"isaac_sim driver requires the [isaac] extra: {exc}. "
            'Install with `pip install -e ".[isaac]" '
            "--extra-index-url https://pypi.nvidia.com`."
        )
        raise ArmDriverError(msg) from exc
    return IsaacSimDriver(arm_cfg)


_DRIVERS.register("isaac_sim", _isaac_sim_factory)


def register_driver(name: str, factory: DriverFactory) -> None:
    """Register a driver class under ``name``."""
    _DRIVERS.register(name, factory)


def get_driver(name: str) -> DriverFactory:
    """Return the driver factory registered under ``name``."""
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
