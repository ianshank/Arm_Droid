"""Robot articulation constants for vendored Isaac Lab task code.

Provides lazy module-level ``SO_ARM100_CFG`` and ``SO_ARM101_CFG``
constants that the vendored ``MuammerBay/isaac_so_arm101`` reach task
code expects (it imports them at module top in
``joint_pos_env_cfg.py``). Both are built via
:func:`armdroid.hardware.isaac_sim.articulation.build_so_arm100_articulation_cfg`
on first reference, with the default :class:`ArmSimIsaacConfig`.

This module exists purely to make the vendored task code work without
modifying it further. Direct armdroid consumers should call
``build_so_arm100_articulation_cfg(sim_cfg, arm_cfg)`` for the
parametrised path; the constants here are equivalent to the function
called with default configs.

Coverage-omit: lazy-imports isaaclab at first reference.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy module-level constants for SO_ARM100/SO_ARM101 ArticulationCfg."""
    if name in ("SO_ARM100_CFG", "SO_ARM101_CFG"):
        # Lazy imports — only triggered when the vendored task code (or
        # an explicit reach-test) accesses these constants.
        from armdroid.config.schema.arm import ArmConfig
        from armdroid.config.schema.sim_isaac import _default_sim_cfg
        from armdroid.hardware.isaac_sim.articulation import (
            build_so_arm100_articulation_cfg,
        )

        sim_cfg = _default_sim_cfg()
        arm_cfg = ArmConfig()
        # Both SO_ARM100 and SO_ARM101 currently resolve to the same
        # default articulation — the SO101-specific calibration override
        # is a TODO follow-up. Document so reviewers don't think it's a bug.
        return build_so_arm100_articulation_cfg(sim_cfg, arm_cfg)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# Note: ``__all__`` deliberately omitted — F822 would flag
# ``SO_ARM100_CFG`` / ``SO_ARM101_CFG`` as undefined because they are
# resolved via ``__getattr__`` on first reference. Vendored task code
# imports the names directly via ``from .robots import SO_ARM100_CFG``
# which triggers ``__getattr__`` correctly.
