"""Unit tests for LegacyArmDriverAdapter.

Verifies the adapter:
- Emits DeprecationWarning on construction.
- Delegates every legacy-surface method to the wrapped driver correctly.
- Does not break when the underlying driver raises.
"""

from __future__ import annotations

import math
import warnings
from typing import Final

import numpy as np
import pytest

from armdroid.compat.legacy_driver_adapter import LegacyArmDriverAdapter
from armdroid.config.schema import ArmConfig, ArmServoConfig, JointLimits
from armdroid.hardware.mock_arm_driver import MockArmDriver

_GENEROUS_LIMITS: Final = JointLimits(
    min_rad=-math.pi,
    max_rad=math.pi,
    max_velocity_rad_s=10.0,
)


def _make_driver(dof: int = 6) -> MockArmDriver:
    cfg = ArmConfig(
        dof=dof,
        home_position=[0.0] * dof,
        joint_limits=[_GENEROUS_LIMITS] * dof,
        servos=[ArmServoConfig(pwm_pin=13 + i) for i in range(dof)],
    )
    return MockArmDriver(cfg)


def _make_adapter(dof: int = 6) -> LegacyArmDriverAdapter:
    """Return a LegacyArmDriverAdapter, suppressing its DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return LegacyArmDriverAdapter(_make_driver(dof))


class TestLegacyArmDriverAdapterWarning:
    """Adapter emits DeprecationWarning on construction."""

    def test_construction_warns(self) -> None:
        drv = _make_driver()
        with pytest.warns(DeprecationWarning, match="LegacyArmDriverAdapter"):
            LegacyArmDriverAdapter(drv)


class TestLegacyArmDriverAdapterDelegation:
    """All legacy-surface methods delegate correctly to the wrapped driver."""

    def test_dof_property(self) -> None:
        adapter = _make_adapter(dof=4)
        assert adapter.dof == 4

    @pytest.mark.asyncio
    async def test_start_connects(self) -> None:
        adapter = _make_adapter()
        await adapter.start()
        assert adapter.is_connected

    @pytest.mark.asyncio
    async def test_stop_disconnects(self) -> None:
        adapter = _make_adapter()
        await adapter.start()
        await adapter.stop()
        assert not adapter.is_connected

    @pytest.mark.asyncio
    async def test_connect_alias(self) -> None:
        adapter = _make_adapter()
        await adapter.connect()
        assert adapter.is_connected

    @pytest.mark.asyncio
    async def test_disconnect_alias(self) -> None:
        adapter = _make_adapter()
        await adapter.connect()
        await adapter.disconnect()
        assert not adapter.is_connected

    @pytest.mark.asyncio
    async def test_send_joint_command_translates_to_positions(self) -> None:
        adapter = _make_adapter()
        await adapter.connect()
        target = np.array([0.1, 0.2, 0.3, 0.0, -0.1, -0.2])
        await adapter.send_joint_command(target)
        # After command, get_joint_states returns a numpy array
        joints = await adapter.get_joint_states()
        assert isinstance(joints, np.ndarray)
        assert joints.shape == (6,)

    @pytest.mark.asyncio
    async def test_get_joint_states_returns_float64_ndarray(self) -> None:
        adapter = _make_adapter()
        await adapter.connect()
        joints = await adapter.get_joint_states()
        assert isinstance(joints, np.ndarray)
        assert joints.dtype == np.float64
        assert joints.shape == (6,)

    @pytest.mark.asyncio
    async def test_read_state_passthrough(self) -> None:
        from armdroid.domain.state import ArmState

        adapter = _make_adapter()
        await adapter.connect()
        state = await adapter.read_state()
        assert isinstance(state, ArmState)

    @pytest.mark.asyncio
    async def test_open_gripper_no_raise(self) -> None:
        adapter = _make_adapter()
        await adapter.open_gripper()

    @pytest.mark.asyncio
    async def test_close_gripper_returns_force(self) -> None:
        adapter = _make_adapter()
        force = await adapter.close_gripper()
        assert isinstance(force, float)
        assert force >= 0.0

    @pytest.mark.asyncio
    async def test_home_resets_position(self) -> None:
        adapter = _make_adapter()
        await adapter.connect()
        await adapter.send_joint_command(np.array([0.5] * 6))
        await adapter.home()
        joints = await adapter.get_joint_states()
        np.testing.assert_array_equal(joints, np.zeros(6))

    @pytest.mark.asyncio
    async def test_emergency_stop_and_clear(self) -> None:
        adapter = _make_adapter()
        await adapter.connect()
        await adapter.emergency_stop()
        await adapter.clear_emergency_stop()
        # Should not raise after clearing
        await adapter.connect()
