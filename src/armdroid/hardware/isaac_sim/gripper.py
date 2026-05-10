"""Pure gripper unit conversion: armdroid normalised [0,1] ↔ URDF radians.

armdroid convention (verified ``domain/state.py`` and
``firmware/arm_esp32/PROTOCOL.md``): ``0=open``, ``1=closed``.
URDF convention from MuammerBay vendor: ``radians_open`` positive
(~1.74), ``radians_closed`` near zero. This module is the **single
source of truth** for the rescale-and-invert; both ``IsaacSimDriver``
and ``SoArmReachIsaacEnv`` import from here. Closes R5 + R7 from PR #8
review.

No isaaclab dep — module is importable on default installs.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from armdroid.config.schema.sim_isaac import ArmSimIsaacConfig


_REL_TOL = 1e-9


def _validate_cfg(cfg: ArmSimIsaacConfig) -> None:
    """Reject degenerate config that would zero-divide the conversion.

    Uses :func:`math.isclose` instead of ``==`` because YAML round-trip
    can introduce sub-LSB float drift (e.g. ``1.74533`` →
    ``1.7453299999...``); literal equality would mask actually-degenerate
    config that happens to drift by a single ULP.
    """
    if math.isclose(
        cfg.gripper_joint_radians_open,
        cfg.gripper_joint_radians_closed,
        rel_tol=_REL_TOL,
    ):
        msg = (
            "ArmSimIsaacConfig.gripper_joint_radians_open == "
            "gripper_joint_radians_closed (within rel_tol=1e-9) — "
            "degenerate gripper range; conversion would divide by zero."
        )
        raise ValueError(msg)
    if math.isclose(
        cfg.gripper_normalised_open,
        cfg.gripper_normalised_closed,
        rel_tol=_REL_TOL,
    ):
        msg = (
            "ArmSimIsaacConfig.gripper_normalised_open == "
            "gripper_normalised_closed (within rel_tol=1e-9) — "
            "degenerate normalised range."
        )
        raise ValueError(msg)


def normalised_to_radians(
    normalised: float,
    cfg: ArmSimIsaacConfig,
) -> float:
    """Map normalised gripper opening (armdroid convention) to URDF radians.

    Args:
        normalised: armdroid normalised value in
            ``[gripper_normalised_open, gripper_normalised_closed]``
            (default ``[0, 1]`` with 0=open, 1=closed).
        cfg: Source of the four conversion fields. Validated each call;
            the cost is two ``math.isclose`` calls — negligible relative
            to the simulation step it precedes.

    Returns:
        URDF gripper joint position (rad).
    """
    _validate_cfg(cfg)
    span_n = cfg.gripper_normalised_closed - cfg.gripper_normalised_open
    span_r = cfg.gripper_joint_radians_closed - cfg.gripper_joint_radians_open
    t = (normalised - cfg.gripper_normalised_open) / span_n
    return cfg.gripper_joint_radians_open + t * span_r


def radians_to_normalised(
    radians: float,
    cfg: ArmSimIsaacConfig,
) -> float:
    """Inverse of :func:`normalised_to_radians`."""
    _validate_cfg(cfg)
    span_n = cfg.gripper_normalised_closed - cfg.gripper_normalised_open
    span_r = cfg.gripper_joint_radians_closed - cfg.gripper_joint_radians_open
    t = (radians - cfg.gripper_joint_radians_open) / span_r
    return cfg.gripper_normalised_open + t * span_n


def normalised_vector_to_radians(
    joints_norm: tuple[float, ...],
    gripper_index: int,
    cfg: ArmSimIsaacConfig,
) -> tuple[float, ...]:
    """Convert one element (the gripper) of a joint vector; pass others through.

    Args:
        joints_norm: Joint vector with the gripper element in armdroid
            normalised units; other elements in radians (or whatever
            units the consumer expects).
        gripper_index: 0-based index of the gripper element. Matches
            ``ArmSimIsaacConfig.gripper_joint_index``.
        cfg: Conversion config.

    Returns:
        Joint vector with only the gripper element rescaled to URDF
        radians; other elements pass through unchanged.

    Raises:
        IndexError: If ``gripper_index`` is out of range.
    """
    if not 0 <= gripper_index < len(joints_norm):
        msg = f"gripper_index {gripper_index} out of range for vector of length {len(joints_norm)}"
        raise IndexError(msg)
    return tuple(
        normalised_to_radians(v, cfg) if i == gripper_index else v
        for i, v in enumerate(joints_norm)
    )


def radians_vector_to_normalised(
    joints_rad: tuple[float, ...],
    gripper_index: int,
    cfg: ArmSimIsaacConfig,
) -> tuple[float, ...]:
    """Inverse of :func:`normalised_vector_to_radians`."""
    if not 0 <= gripper_index < len(joints_rad):
        msg = f"gripper_index {gripper_index} out of range for vector of length {len(joints_rad)}"
        raise IndexError(msg)
    return tuple(
        radians_to_normalised(v, cfg) if i == gripper_index else v for i, v in enumerate(joints_rad)
    )


__all__ = [
    "normalised_to_radians",
    "normalised_vector_to_radians",
    "radians_to_normalised",
    "radians_vector_to_normalised",
]
