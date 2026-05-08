"""Modern-surface tests for the extended :class:`MockArmDriver`.

Cover the lifecycle (``connect`` / ``disconnect`` / ``is_connected``),
interpolated motion (``send_joint_positions`` / ``read_state``), latched
e-stop, joint-limit and velocity-limit validation, and concurrency under
``asyncio.gather``.

The legacy adapter surface (``send_joint_command`` / ``home`` /
``open_gripper`` / ``close_gripper`` / ``start`` / ``stop`` /
``get_joint_states``) is covered by the existing
``test_mock_arm_driver.py``.
"""

from __future__ import annotations

import asyncio
import math
from typing import Final

import numpy as np
import pytest

from armdroid.config.schema import (
    ArmConfig,
    ArmServoConfig,
    JointLimits,
)
from armdroid.hardware.mock_arm_driver import MockArmDriver
from armdroid.protocols import (
    ArmCommandRejected,
    ArmDriverError,
    ArmDriverProtocol,
    ArmState,
)

_GENEROUS_LIMITS: Final = JointLimits(
    min_rad=-math.pi,
    max_rad=math.pi,
    max_velocity_rad_s=10.0,
)


def _make_cfg(dof: int = 6) -> ArmConfig:
    return ArmConfig(
        dof=dof,
        home_position=[0.0] * dof,
        joint_limits=[_GENEROUS_LIMITS] * dof,
        servos=[ArmServoConfig(pwm_pin=13 + i) for i in range(dof)],
    )


class TestProtocolConformance:
    """The mock satisfies the runtime-checkable Protocol."""

    def test_satisfies_protocol(self) -> None:
        driver = MockArmDriver(_make_cfg())
        assert isinstance(driver, ArmDriverProtocol)


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_connect_disconnect_idempotent(self) -> None:
        driver = MockArmDriver(_make_cfg())
        assert not driver.is_connected
        await driver.connect()
        await driver.connect()  # no-op
        assert driver.is_connected
        await driver.disconnect()
        await driver.disconnect()  # no-op
        assert not driver.is_connected

    @pytest.mark.asyncio
    async def test_modern_command_before_connect_raises(self) -> None:
        driver = MockArmDriver(_make_cfg())
        with pytest.raises(ArmDriverError):
            await driver.send_joint_positions((0.0,) * 6, duration_s=1.0)

    @pytest.mark.asyncio
    async def test_modern_read_state_before_connect_raises(self) -> None:
        driver = MockArmDriver(_make_cfg())
        with pytest.raises(ArmDriverError):
            await driver.read_state()


class TestValidation:
    """The modern surface rejects every contract violation."""

    @pytest.mark.asyncio
    async def test_wrong_length_rejected(self) -> None:
        driver = MockArmDriver(_make_cfg())
        await driver.connect()
        with pytest.raises(ArmCommandRejected, match="Expected 6"):
            await driver.send_joint_positions((0.0, 0.0, 0.0), duration_s=1.0)

    @pytest.mark.asyncio
    async def test_non_finite_rejected(self) -> None:
        driver = MockArmDriver(_make_cfg())
        await driver.connect()
        bad = (0.0, math.nan, 0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ArmCommandRejected, match="non-finite"):
            await driver.send_joint_positions(bad, duration_s=1.0)

    @pytest.mark.asyncio
    async def test_inf_rejected(self) -> None:
        driver = MockArmDriver(_make_cfg())
        await driver.connect()
        bad = (math.inf, 0.0, 0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ArmCommandRejected, match="non-finite"):
            await driver.send_joint_positions(bad, duration_s=1.0)

    @pytest.mark.asyncio
    async def test_out_of_limit_rejected(self) -> None:
        cfg = _make_cfg()
        # tighter limit just for joint 0 to make a value outside it
        cfg.joint_limits[0] = JointLimits(min_rad=-1.0, max_rad=1.0, max_velocity_rad_s=10.0)
        driver = MockArmDriver(cfg)
        await driver.connect()
        bad = (2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ArmCommandRejected, match=r"joint\[0\]"):
            await driver.send_joint_positions(bad, duration_s=1.0)

    @pytest.mark.asyncio
    async def test_zero_duration_rejected(self) -> None:
        driver = MockArmDriver(_make_cfg())
        await driver.connect()
        with pytest.raises(ArmCommandRejected, match="duration_s"):
            await driver.send_joint_positions((0.0,) * 6, duration_s=0.0)

    @pytest.mark.asyncio
    async def test_negative_duration_rejected(self) -> None:
        driver = MockArmDriver(_make_cfg())
        await driver.connect()
        with pytest.raises(ArmCommandRejected, match="duration_s"):
            await driver.send_joint_positions((0.0,) * 6, duration_s=-1.0)

    @pytest.mark.asyncio
    async def test_velocity_limit_violation_rejected(self) -> None:
        """1 rad in 0.05 s = 20 rad/s, above the 10 rad/s limit."""
        driver = MockArmDriver(_make_cfg())
        await driver.connect()
        target = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ArmCommandRejected, match="rad/s"):
            await driver.send_joint_positions(target, duration_s=0.05)


class TestInterpolation:
    """Read-state returns a position consistent with elapsed time."""

    @pytest.mark.asyncio
    async def test_state_interpolates_halfway(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_now = [1000.0]
        monkeypatch.setattr(
            "armdroid.hardware.mock_arm_driver.time.monotonic",
            lambda: fake_now[0],
        )
        driver = MockArmDriver(_make_cfg())
        await driver.connect()
        target = (1.0, 0.5, -0.5, 0.0, 0.0, 0.0)
        await driver.send_joint_positions(target, duration_s=2.0)

        fake_now[0] += 1.0  # halfway
        state = await driver.read_state()
        assert state.is_moving
        assert state.joint_positions[0] == pytest.approx(0.5)
        assert state.joint_positions[1] == pytest.approx(0.25)
        assert state.joint_positions[2] == pytest.approx(-0.25)

        fake_now[0] += 1.5  # past end
        state2 = await driver.read_state()
        assert not state2.is_moving
        assert state2.joint_positions == target
        assert all(v == 0.0 for v in state2.joint_velocities)

    @pytest.mark.asyncio
    async def test_chained_command_starts_from_current(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_now = [1000.0]
        monkeypatch.setattr(
            "armdroid.hardware.mock_arm_driver.time.monotonic",
            lambda: fake_now[0],
        )
        driver = MockArmDriver(_make_cfg())
        await driver.connect()
        first = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        await driver.send_joint_positions(first, duration_s=2.0)
        fake_now[0] += 1.0  # halfway
        second = (1.5, 0.0, 0.0, 0.0, 0.0, 0.0)
        await driver.send_joint_positions(second, duration_s=1.0)
        state = await driver.read_state()
        assert state.joint_positions[0] == pytest.approx(0.5, abs=1e-6)

    @pytest.mark.asyncio
    async def test_velocity_check_uses_interpolated_start(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Velocity is checked against the *interpolated* position at command
        time, not the previous segment target.  A chained command that would
        be safe from the prior *target* but too fast from the current
        *interpolated* position must be rejected.

        Setup: start from 0.0, issue a slow 2 s move to 0.2 rad.  After
        1 s the arm is at 0.1 rad.  Now issue a second move of 0.9 rad
        (to 1.0) in 0.05 s: that requires 0.9 / 0.05 = 18 rad/s, well above
        the 10 rad/s limit, even though the target gap (1.0 - 0.2 = 0.8)
        would only require 16 rad/s from the *target*.
        """
        fake_now = [1000.0]
        monkeypatch.setattr(
            "armdroid.hardware.mock_arm_driver.time.monotonic",
            lambda: fake_now[0],
        )
        cfg = _make_cfg()  # max_velocity_rad_s = 10.0
        driver = MockArmDriver(cfg)
        await driver.connect()

        # First command: 0 -> 0.2 over 2 s (0.1 rad/s — well within limit)
        await driver.send_joint_positions((0.2,) + (0.0,) * 5, duration_s=2.0)

        # Advance to half-way: interpolated j0 = 0.1 rad
        fake_now[0] += 1.0

        # Second command: current 0.1 -> 1.0 in 0.05 s = 18 rad/s > 10.0 limit
        with pytest.raises(ArmCommandRejected, match="rad/s"):
            await driver.send_joint_positions((1.0,) + (0.0,) * 5, duration_s=0.05)


class TestEmergencyStop:
    """Latch freezes the arm and rejects subsequent motion commands."""

    @pytest.mark.asyncio
    async def test_estop_freezes_and_rejects(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_now = [1000.0]
        monkeypatch.setattr(
            "armdroid.hardware.mock_arm_driver.time.monotonic",
            lambda: fake_now[0],
        )
        driver = MockArmDriver(_make_cfg())
        await driver.connect()
        target = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        await driver.send_joint_positions(target, duration_s=2.0)
        fake_now[0] += 1.0  # mid-motion
        await driver.emergency_stop()

        state = await driver.read_state()
        assert state.estop_active
        assert not state.is_moving
        frozen_pos = state.joint_positions[0]
        assert 0.0 < frozen_pos < 1.0

        # Time advances but the arm doesn't move
        fake_now[0] += 5.0
        state2 = await driver.read_state()
        assert state2.joint_positions[0] == pytest.approx(frozen_pos)

        # Motion commands rejected while latched
        with pytest.raises(ArmCommandRejected, match="e-stop"):
            await driver.send_joint_positions((0.0,) * 6, duration_s=1.0)

        # Cleared
        await driver.clear_emergency_stop()
        state3 = await driver.read_state()
        assert not state3.estop_active

        # Now a new command works
        await driver.send_joint_positions((0.0,) * 6, duration_s=1.0)


class TestConcurrency:
    """Read/send under asyncio.gather is safe (no torn segment state)."""

    @pytest.mark.asyncio
    async def test_concurrent_send_and_read(self) -> None:
        driver = MockArmDriver(_make_cfg())
        await driver.connect()

        async def reader() -> None:
            for _ in range(50):
                state = await driver.read_state()
                assert isinstance(state, ArmState)
                assert len(state.joint_positions) == 6
                await asyncio.sleep(0)

        async def writer() -> None:
            for i in range(50):
                target = ((i % 10) * 0.1,) * 6
                await driver.send_joint_positions(target, duration_s=0.5)
                await asyncio.sleep(0)

        await asyncio.gather(reader(), writer())


class TestLegacyAdapterCompat:
    """Legacy methods route through the modern surface but preserve old UX."""

    @pytest.mark.asyncio
    async def test_get_joint_states_returns_ndarray(self) -> None:
        driver = MockArmDriver(_make_cfg())
        joints = await driver.get_joint_states()
        assert isinstance(joints, np.ndarray)
        assert joints.dtype == np.float64
        assert joints.shape == (6,)

    @pytest.mark.asyncio
    async def test_send_joint_command_clips_silently(self) -> None:
        # Tighten joint 0 limits so the legacy clip is observable
        cfg = _make_cfg()
        cfg.joint_limits[0] = JointLimits(min_rad=-0.5, max_rad=0.5, max_velocity_rad_s=10.0)
        driver = MockArmDriver(cfg)
        # Out-of-limit angle should be clipped silently to joint 0's max_rad
        target = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        await driver.send_joint_command(target)
        joints = await driver.get_joint_states()
        assert joints[0] == pytest.approx(0.5, abs=1e-6)

    @pytest.mark.asyncio
    async def test_send_joint_command_wrong_length_raises_value_error(self) -> None:
        driver = MockArmDriver(_make_cfg())
        with pytest.raises(ValueError, match="Expected 6"):
            await driver.send_joint_command(np.zeros(4))

    @pytest.mark.asyncio
    async def test_legacy_start_stop_proxy_to_lifecycle(self) -> None:
        driver = MockArmDriver(_make_cfg())
        await driver.start()
        assert driver.is_connected
        await driver.stop()
        assert not driver.is_connected

    @pytest.mark.asyncio
    async def test_home_clears_estop_silently(self) -> None:
        """Legacy callers don't know about estop — home() must always work."""
        driver = MockArmDriver(_make_cfg())
        await driver.connect()
        await driver.emergency_stop()
        await driver.home()  # should not raise
        joints = await driver.get_joint_states()
        np.testing.assert_array_equal(joints, np.zeros(6))
