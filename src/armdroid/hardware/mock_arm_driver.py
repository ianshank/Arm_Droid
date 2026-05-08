"""In-memory mock arm driver implementing :class:`ArmDriverProtocol`.

Default driver for CI and any environment with ``cfg.mock_hardware=True``.
Performs first-order linear interpolation between commanded poses, enforces
the same per-joint limits the real :class:`Esp32JsonDriver` enforces, and
supports the latched e-stop. Designed to be deterministic so tests can
freeze ``time.monotonic`` and assert on interpolated positions.

Both the modern surface (``connect`` / ``read_state`` /
``send_joint_positions`` / ``emergency_stop`` / ``clear_emergency_stop``)
and the legacy adapters (``start`` / ``send_joint_command`` /
``get_joint_states`` / ``open_gripper`` / ``close_gripper`` / ``home``) are
implemented so existing controllers, primitives, and tests continue to
work unchanged after the protocol extension.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from armdroid.logging.setup import get_logger
from armdroid.protocols import ArmCommandRejected, ArmDriverError, ArmState

if TYPE_CHECKING:
    from armdroid.config.schema import ArmConfig

_log = get_logger(__name__)

# Default grip force returned by close_gripper when the gripper transitions
# from open -> closed. Distinct from joint position; no equivalent in the
# modern protocol but kept for backwards compatibility with controllers
# written against the legacy surface.
_MOCK_GRIP_FORCE_N = 1.0


class MockArmDriver:
    """In-memory arm simulator satisfying :class:`ArmDriverProtocol`.

    Joint state advances linearly between successive commanded poses.
    Each call to :meth:`read_state` recomputes the position the arm would
    be at given the elapsed wall-clock time since the last command.

    Args:
        cfg: Validated :class:`ArmConfig`. Joint limits and home pose are
            honoured; transport settings are ignored (this is a mock).
    """

    def __init__(self, cfg: ArmConfig) -> None:
        """Initialise the mock at the configured home position.

        Args:
            cfg: Arm configuration with dof, home_position, joint_limits,
                home_duration_s.
        """
        self._cfg = cfg
        self._dof = cfg.dof
        self._connected = False
        self._estop = False
        self._gripper_open = True  # legacy gripper toggle (cfg.dof == 6 path)

        # Current commanded segment: arm interpolates from
        # ``_segment_start`` to ``_segment_target`` over ``_segment_duration_s``,
        # starting at wall-clock ``_segment_start_time_s``.
        home = tuple(float(v) for v in cfg.home_position)
        self._segment_start: tuple[float, ...] = home
        self._segment_target: tuple[float, ...] = home
        self._segment_duration_s: float = 0.0
        self._segment_start_time_s: float = time.monotonic()

        # Asyncio lock guards segment state so interleaved
        # send_joint_positions / read_state under asyncio.gather don't tear.
        self._lock = asyncio.Lock()
        _log.info("mock_arm_driver_init", dof=self._dof)

    # ------------------------------------------------------------------ #
    # Modern lifecycle
    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        """Mark the mock as connected. Idempotent."""
        if self._connected:
            return
        self._connected = True
        _log.info("mock_arm_connected", home=list(self._segment_start))

    async def disconnect(self) -> None:
        """Mark the mock as disconnected. Idempotent."""
        if not self._connected:
            return
        self._connected = False
        _log.info("mock_arm_disconnected")

    @property
    def is_connected(self) -> bool:
        """Whether the mock has been ``connect()``-ed."""
        return self._connected

    # ------------------------------------------------------------------ #
    # Modern motion
    # ------------------------------------------------------------------ #

    async def send_joint_positions(
        self,
        positions: tuple[float, ...],
        duration_s: float,
    ) -> None:
        """Validate and stage an interpolated move."""
        self._require_connected()
        self._validate_command(positions, duration_s)

        async with self._lock:
            if self._estop:
                msg = "Cannot command motion while e-stop is latched"
                raise ArmCommandRejected(msg)

            now = time.monotonic()
            current = self._interpolate_unlocked(now)
            self._segment_start = current
            self._segment_target = positions
            self._segment_duration_s = duration_s
            self._segment_start_time_s = now

        _log.debug("mock_arm_command", target=list(positions), duration_s=duration_s)

    async def read_state(self) -> ArmState:
        """Compute and return the current interpolated state."""
        self._require_connected()
        async with self._lock:
            now = time.monotonic()
            position = self._interpolate_unlocked(now)
            elapsed = now - self._segment_start_time_s
            is_moving = (
                self._segment_duration_s > 0.0
                and elapsed < self._segment_duration_s
                and not self._estop
            )
            velocity = self._velocity_unlocked(is_moving)
            return ArmState(
                joint_positions=position,
                joint_velocities=velocity,
                is_moving=is_moving,
                estop_active=self._estop,
                timestamp_s=now,
            )

    # ------------------------------------------------------------------ #
    # Modern safety
    # ------------------------------------------------------------------ #

    async def emergency_stop(self) -> None:
        """Latch e-stop and freeze the arm at the current pose."""
        async with self._lock:
            now = time.monotonic()
            frozen = self._interpolate_unlocked(now)
            self._segment_start = frozen
            self._segment_target = frozen
            self._segment_duration_s = 0.0
            self._segment_start_time_s = now
            self._estop = True
        _log.warning("mock_arm_estop_latched", frozen_at=list(frozen))

    async def clear_emergency_stop(self) -> None:
        """Release the e-stop latch."""
        async with self._lock:
            self._estop = False
        _log.info("mock_arm_estop_cleared")

    # ------------------------------------------------------------------ #
    # Legacy adapters — preserved for backwards compatibility.
    # ------------------------------------------------------------------ #

    @property
    def dof(self) -> int:
        """Joint vector length (legacy)."""
        return self._dof

    async def start(self) -> None:
        """Legacy lifecycle alias — calls :meth:`connect`."""
        await self.connect()

    async def stop(self) -> None:
        """Legacy lifecycle alias — calls :meth:`disconnect`."""
        await self.disconnect()

    async def get_joint_states(self) -> NDArray[np.float64]:
        """Legacy state read — returns numpy array of joint positions.

        Auto-connects on first call to preserve the historical behaviour
        where legacy controllers did not have to think about lifecycle.
        """
        if not self._connected:
            await self.connect()
        state = await self.read_state()
        return np.array(state.joint_positions, dtype=np.float64)

    async def send_joint_command(self, target_angles: NDArray[np.float64]) -> None:
        """Legacy step command — instantaneous, silent per-joint clipping.

        Preserves the original SO-ARM100 mock semantics: out-of-range angles
        are silently clipped rather than rejected, the move completes
        immediately (no interpolation duration), and velocity limits are
        not enforced. Modern callers should use :meth:`send_joint_positions`
        which validates strictly.

        Raises:
            ValueError: only when the input length doesn't match dof.
        """
        if not self._connected:
            await self.connect()
        if len(target_angles) != self._dof:
            msg = f"Expected {self._dof} joint angles, got {len(target_angles)}"
            raise ValueError(msg)
        # Silent per-joint clipping — preserves the historical behaviour
        # where out-of-range angles were np.clip()'d into ±π.
        clipped = tuple(
            max(self._cfg.joint_limits[i].min_rad, min(self._cfg.joint_limits[i].max_rad, float(v)))
            for i, v in enumerate(target_angles)
        )
        await self._set_state_instantly(clipped)

    async def close_gripper(self) -> float:
        """Legacy gripper close — returns simulated grip force.

        On a 6-DoF arm without a protocol gripper joint, this is purely a
        bookkeeping flag. Commit 7 propagates the gripper to joint index
        ``dof - 1`` of :meth:`send_joint_positions`.
        """
        if not self._connected:
            await self.connect()
        if self._gripper_open:
            self._gripper_open = False
            grip_force = _MOCK_GRIP_FORCE_N
        else:
            grip_force = 0.0
        _log.debug("mock_gripper_close", force=grip_force)
        return grip_force

    async def open_gripper(self) -> None:
        """Legacy gripper open."""
        if not self._connected:
            await self.connect()
        self._gripper_open = True
        _log.debug("mock_gripper_open")

    async def home(self) -> None:
        """Move arm to home position — instantaneous, like the legacy mock.

        The original SO-ARM100 driver's mock had no notion of motion
        duration; ``home()`` snapped to the home pose immediately. We keep
        that semantics here so existing tests (and any controller that
        polls joints right after ``home()``) continue to work.
        """
        if not self._connected:
            await self.connect()
        home = tuple(float(v) for v in self._cfg.home_position)
        # Clear any latched e-stop so home() always succeeds (matches the
        # historical behaviour — the legacy mock had no e-stop concept).
        if self._estop:
            await self.clear_emergency_stop()
        await self._set_state_instantly(home)
        self._gripper_open = True
        _log.info("mock_arm_homed")

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _require_connected(self) -> None:
        if not self._connected:
            msg = "MockArmDriver is not connected"
            raise ArmDriverError(msg)

    def _validate_command(
        self,
        positions: tuple[float, ...],
        duration_s: float,
    ) -> None:
        if len(positions) != self._dof:
            msg = f"Expected {self._dof} joint positions, got {len(positions)}"
            raise ArmCommandRejected(msg)
        if duration_s <= 0.0:
            msg = f"duration_s must be positive, got {duration_s}"
            raise ArmCommandRejected(msg)
        for idx, value in enumerate(positions):
            if not math.isfinite(value):
                msg = f"joint[{idx}] is non-finite: {value}"
                raise ArmCommandRejected(msg)
            limits = self._cfg.joint_limits[idx]
            if not (limits.min_rad <= value <= limits.max_rad):
                msg = f"joint[{idx}]={value} outside [{limits.min_rad}, {limits.max_rad}]"
                raise ArmCommandRejected(msg)
        # Velocity feasibility — reject moves that would exceed any joint's
        # max velocity. Compared against the *previous* segment_target so
        # the user cannot circumvent the limit by chaining commands.
        for idx, (start, target) in enumerate(zip(self._segment_target, positions, strict=True)):
            required_speed = abs(target - start) / duration_s
            limit = self._cfg.joint_limits[idx].max_velocity_rad_s
            if required_speed > limit:
                msg = (
                    f"joint[{idx}] would need {required_speed:.3f} rad/s, "
                    f"limit is {limit:.3f} rad/s"
                )
                raise ArmCommandRejected(msg)

    async def _set_state_instantly(self, positions: tuple[float, ...]) -> None:
        """Snap the simulated arm to ``positions`` with no interpolation.

        Used by the legacy adapters (``send_joint_command``, ``home``) to
        preserve their historical "set and stick" semantics. Bypasses
        velocity validation; the caller is expected to have already
        clipped per-joint limits if needed.
        """
        async with self._lock:
            now = time.monotonic()
            self._segment_start = positions
            self._segment_target = positions
            self._segment_duration_s = 0.0
            self._segment_start_time_s = now

    def _interpolate_unlocked(self, now: float) -> tuple[float, ...]:
        """Linear interpolation between segment_start and segment_target.

        Caller must hold ``self._lock``.
        """
        if self._segment_duration_s <= 0.0:
            return self._segment_target
        elapsed = now - self._segment_start_time_s
        if elapsed >= self._segment_duration_s:
            return self._segment_target
        alpha = elapsed / self._segment_duration_s
        return tuple(
            start + alpha * (target - start)
            for start, target in zip(self._segment_start, self._segment_target, strict=True)
        )

    def _velocity_unlocked(self, is_moving: bool) -> tuple[float, ...]:
        """Constant velocity during a segment, zero otherwise."""
        if not is_moving or self._segment_duration_s <= 0.0:
            return tuple(0.0 for _ in range(self._dof))
        return tuple(
            (target - start) / self._segment_duration_s
            for start, target in zip(self._segment_start, self._segment_target, strict=True)
        )
