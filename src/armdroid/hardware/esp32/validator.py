"""Local command validation for the ESP32 JSON driver."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from armdroid.domain.errors import ArmCommandRejected
from armdroid.domain.state import ArmState

if TYPE_CHECKING:
    from armdroid.config.schema.arm import JointLimits

# Velocity-check anchor resolution order (informational, not executable).
# Documents the priority order used by velocity_anchor().
_VELOCITY_CHECK_ANCHOR_FALLBACK_ORDER: tuple[str, str, str] = (
    "observed_state",
    "last_commanded",
    "home_position",
)


def velocity_anchor(
    latest_state: ArmState | None,
    last_commanded_target: tuple[float, ...] | None,
    home_position: list[float],
) -> tuple[float, ...]:
    """Return the best available start-position for velocity checks.

    Resolution order:

    1. Most recently *observed* state (latest heartbeat).
    2. Last successfully commanded target (post-ack).
    3. Configured home position.

    Args:
        latest_state: Most recently cached firmware state, or ``None``.
        last_commanded_target: Last position that received a firmware ack,
            or ``None``.
        home_position: Configured home position from the arm config.

    Returns:
        Best available joint-position tuple for velocity feasibility checks.
    """
    if latest_state is not None:
        return latest_state.joint_positions
    if last_commanded_target is not None:
        return last_commanded_target
    return tuple(float(v) for v in home_position)


def validate_joint_positions(
    positions: tuple[float, ...],
    duration_s: float,
    dof: int,
    joint_limits: list[JointLimits],
    anchor: tuple[float, ...],
) -> None:
    """Validate a joint position command locally before writing to the wire.

    Checks performed:

    * Joint count matches the configured DoF.
    * Duration is strictly positive.
    * All values are finite.
    * All values are within their per-joint ``[min_rad, max_rad]`` range.
    * The required velocity for each joint does not exceed
      ``max_velocity_rad_s`` given the provided *anchor* start position.

    Args:
        positions: Target joint positions (radians).
        duration_s: Time budget for the move.
        dof: Expected joint count.
        joint_limits: Per-joint limit configuration objects.
        anchor: Start position used for velocity feasibility checks.

    Raises:
        ArmCommandRejected: If any validation check fails.
    """
    if len(positions) != dof:
        msg = f"Expected {dof} joint positions, got {len(positions)}"
        raise ArmCommandRejected(msg)
    if duration_s <= 0.0:
        msg = f"duration_s must be positive, got {duration_s}"
        raise ArmCommandRejected(msg)
    for idx, value in enumerate(positions):
        if not math.isfinite(value):
            msg = f"joint[{idx}] is non-finite: {value}"
            raise ArmCommandRejected(msg)
        limits = joint_limits[idx]
        if not (limits.min_rad <= value <= limits.max_rad):
            msg = f"joint[{idx}]={value} outside [{limits.min_rad}, {limits.max_rad}]"
            raise ArmCommandRejected(msg)
    for idx, (s, t) in enumerate(zip(anchor, positions, strict=True)):
        required_speed = abs(t - s) / duration_s
        limit = joint_limits[idx].max_velocity_rad_s
        if required_speed > limit:
            msg = f"joint[{idx}] would need {required_speed:.3f} rad/s, limit is {limit:.3f} rad/s"
            raise ArmCommandRejected(msg)


__all__ = [
    "_VELOCITY_CHECK_ANCHOR_FALLBACK_ORDER",
    "validate_joint_positions",
    "velocity_anchor",
]
