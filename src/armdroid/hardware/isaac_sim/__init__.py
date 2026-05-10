"""Isaac Sim 5.1 / Isaac Lab 2.3 driver package (PR-B).

Module top-level deliberately does NOT import isaaclab, torch, or any
runtime dep beyond the pure-Python sub-modules. Lazy imports happen
inside ``IsaacSimDriver.connect()``.

Re-exports:
    IsaacSimDriver: ArmDriverProtocol-conformant driver. Requires the
        [isaac] extra at runtime; module imports without it.
    gripper: Pure conversion functions (no isaaclab dep). Single source
        of truth for the URDF radians ↔ armdroid normalised
        rescale-and-invert (closes R5 + R7).
    build_so_arm100_articulation_cfg: ArticulationCfg builder.
        Lazy-imports isaaclab inside the function body.
"""

from __future__ import annotations

from armdroid.hardware.isaac_sim.gripper import (
    normalised_to_radians,
    normalised_vector_to_radians,
    radians_to_normalised,
    radians_vector_to_normalised,
)

__all__ = [
    "normalised_to_radians",
    "normalised_vector_to_radians",
    "radians_to_normalised",
    "radians_vector_to_normalised",
]


def __getattr__(name: str) -> object:
    """Lazy import for IsaacSimDriver + articulation builder.

    Importing the driver / articulation eagerly would force isaaclab to
    load even when the [isaac] extra isn't installed (since the driver
    catches the ImportError inside connect(), not at module top). The
    lazy attribute hook keeps the package importable on default
    installs and only triggers the heavy import at first reference.
    """
    if name == "IsaacSimDriver":
        from armdroid.hardware.isaac_sim.driver import IsaacSimDriver

        return IsaacSimDriver
    if name == "build_so_arm100_articulation_cfg":
        from armdroid.hardware.isaac_sim.articulation import (
            build_so_arm100_articulation_cfg,
        )

        return build_so_arm100_articulation_cfg
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
