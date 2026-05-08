"""Hypothesis property-based tests for arm-driver invariants.

These tests run the modern protocol surface against random valid inputs
and assert structural invariants that should hold regardless of the
specific values:

* `send_joint_positions` either returns cleanly or raises one of the
  documented exception types.
* `read_state` always returns an `ArmState` with the configured number
  of joints.
* `emergency_stop` always succeeds and is reflected in the next
  `read_state`.
* JSON encoding / decoding round-trip for the wire format never loses
  joint values within float precision.
"""

from __future__ import annotations

import contextlib
import json
import math
from typing import Final

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from armdroid.config.schema import ArmConfig, ArmServoConfig, JointLimits
from armdroid.hardware.mock_arm_driver import MockArmDriver
from armdroid.protocols import (
    ArmCommandRejected,
    ArmDriverError,
    ArmState,
)

_GENEROUS_LIMITS: Final = JointLimits(
    min_rad=-math.pi,
    max_rad=math.pi,
    max_velocity_rad_s=10.0,
)


def _cfg() -> ArmConfig:
    return ArmConfig(
        dof=6,
        home_position=[0.0] * 6,
        joint_limits=[_GENEROUS_LIMITS] * 6,
        servos=[ArmServoConfig(pwm_pin=13 + i) for i in range(6)],
    )


@st.composite
def joint_vectors(draw: st.DrawFn) -> tuple[float, ...]:
    """Tuples of 6 floats spanning the generous-limits range."""
    return tuple(draw(st.floats(min_value=-math.pi, max_value=math.pi)) for _ in range(6))


class TestSendJointPositionsContract:
    @given(positions=joint_vectors(), duration_s=st.floats(min_value=0.5, max_value=5.0))
    @settings(max_examples=30, deadline=None)
    @pytest.mark.asyncio
    async def test_valid_input_succeeds_or_raises_documented(
        self,
        positions: tuple[float, ...],
        duration_s: float,
    ) -> None:
        """Valid input either succeeds or raises ArmCommandRejected/Error."""
        driver = MockArmDriver(_cfg())
        await driver.connect()
        try:
            with contextlib.suppress(ArmCommandRejected, ArmDriverError):
                await driver.send_joint_positions(positions, duration_s)
        finally:
            await driver.disconnect()


class TestReadStateShape:
    @given(positions=joint_vectors(), duration_s=st.floats(min_value=0.5, max_value=5.0))
    @settings(max_examples=20, deadline=None)
    @pytest.mark.asyncio
    async def test_read_state_always_returns_correct_shape(
        self,
        positions: tuple[float, ...],
        duration_s: float,
    ) -> None:
        driver = MockArmDriver(_cfg())
        await driver.connect()
        try:
            with contextlib.suppress(ArmCommandRejected):
                await driver.send_joint_positions(positions, duration_s)
            state = await driver.read_state()
            assert isinstance(state, ArmState)
            assert len(state.joint_positions) == 6
            assert len(state.joint_velocities) == 6
        finally:
            await driver.disconnect()


class TestEstopAlwaysReachable:
    @given(positions=joint_vectors(), duration_s=st.floats(min_value=0.5, max_value=5.0))
    @settings(max_examples=15, deadline=None)
    @pytest.mark.asyncio
    async def test_estop_always_succeeds(
        self,
        positions: tuple[float, ...],
        duration_s: float,
    ) -> None:
        driver = MockArmDriver(_cfg())
        await driver.connect()
        try:
            with contextlib.suppress(ArmCommandRejected):
                await driver.send_joint_positions(positions, duration_s)
            await driver.emergency_stop()
            state = await driver.read_state()
            assert state.estop_active
        finally:
            await driver.disconnect()


class TestJsonRoundTrip:
    """Wire-format encoding round-trip preserves joint values."""

    @given(joint_vectors())
    @settings(max_examples=50, deadline=None)
    def test_joint_vector_round_trips_via_json(
        self,
        positions: tuple[float, ...],
    ) -> None:
        msg = {
            "t": "cmd",
            "id": 1,
            "ts": 0.0,
            "cmd": "set_joints",
            "q": list(positions),
            "dur_ms": 1000,
        }
        encoded = json.dumps(msg, separators=(",", ":"))
        decoded = json.loads(encoded)
        for original, restored in zip(positions, decoded["q"], strict=True):
            assert original == pytest.approx(restored, rel=1e-9)
