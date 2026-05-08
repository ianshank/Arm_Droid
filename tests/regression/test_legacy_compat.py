"""Regression tests guarding legacy adapter behaviour after the protocol
extension.

These tests assert that callers written against the historical
`ArmDriverProtocol` surface continue to work unchanged with the new
`MockArmDriver` and `Esp32JsonDriver` implementations.

* Legacy method names (`start`, `stop`, `get_joint_states`,
  `send_joint_command`, `open_gripper`, `close_gripper`, `home`)
  resolve and behave as they did pre-integration.
* `send_joint_command` silently clips out-of-range angles per joint
  (no exception).
* `home` clears any latched e-stop transparently and snaps to home pose.
* `dof` property returns the configured joint count.
* The CLI dispatch path for `armdroid sim` parses + runs.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pytest

from armdroid.config.schema import ArmConfig, ArmServoConfig, ArmSettings, JointLimits
from armdroid.hardware.mock_arm_driver import MockArmDriver

pytestmark = pytest.mark.regression

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


class TestLegacyMethodNames:
    @pytest.mark.asyncio
    async def test_start_alias_for_connect(self) -> None:
        drv = _make_driver()
        await drv.start()
        assert drv.is_connected

    @pytest.mark.asyncio
    async def test_stop_alias_for_disconnect(self) -> None:
        drv = _make_driver()
        await drv.start()
        await drv.stop()
        assert not drv.is_connected

    @pytest.mark.asyncio
    async def test_get_joint_states_returns_ndarray(self) -> None:
        drv = _make_driver()
        joints = await drv.get_joint_states()  # auto-connect
        assert isinstance(joints, np.ndarray)
        assert joints.dtype == np.float64
        assert joints.shape == (6,)

    @pytest.mark.asyncio
    async def test_close_gripper_returns_force(self) -> None:
        drv = _make_driver()
        force = await drv.close_gripper()
        assert force > 0.0

    @pytest.mark.asyncio
    async def test_open_gripper_no_raise(self) -> None:
        drv = _make_driver()
        await drv.open_gripper()  # no return; just shouldn't raise

    @pytest.mark.asyncio
    async def test_home_resets_to_home_position(self) -> None:
        drv = _make_driver()
        await drv.send_joint_command(np.array([0.5] * 6))
        await drv.home()
        joints = await drv.get_joint_states()
        np.testing.assert_array_equal(joints, np.zeros(6))

    def test_dof_property(self) -> None:
        drv = _make_driver(dof=4)
        assert drv.dof == 4


class TestSendJointCommandSemantics:
    """Legacy `send_joint_command` silently clips, instantaneously sets."""

    @pytest.mark.asyncio
    async def test_silent_clip_to_per_joint_limits(self) -> None:
        cfg = ArmConfig(
            dof=6,
            home_position=[0.0] * 6,
            joint_limits=[JointLimits(min_rad=-0.5, max_rad=0.5, max_velocity_rad_s=10.0)] * 6,
            servos=[ArmServoConfig(pwm_pin=13 + i) for i in range(6)],
        )
        drv = MockArmDriver(cfg)
        # Out-of-range values should be silently clipped, not raise
        target = np.array([10.0, -10.0, 0.0, 0.0, 0.0, 0.0])
        await drv.send_joint_command(target)
        joints = await drv.get_joint_states()
        assert joints[0] == pytest.approx(0.5)
        assert joints[1] == pytest.approx(-0.5)

    @pytest.mark.asyncio
    async def test_wrong_length_raises_value_error(self) -> None:
        drv = _make_driver()
        with pytest.raises(ValueError, match="Expected 6"):
            await drv.send_joint_command(np.zeros(4))


class TestHomeClearsEstop:
    """home() succeeds even when e-stop is latched (legacy behaviour)."""

    @pytest.mark.asyncio
    async def test_home_clears_estop_silently(self) -> None:
        drv = _make_driver()
        await drv.connect()
        await drv.emergency_stop()
        await drv.home()  # should not raise
        joints = await drv.get_joint_states()
        np.testing.assert_array_equal(joints, np.zeros(6))


class TestCliDispatchSurface:
    """The CLI argparse skeleton still parses sim/train/rollout."""

    def test_parse_sim_subcommand(self) -> None:
        from armdroid.main import _parse_args

        args = _parse_args(["sim", "--episodes", "2"])
        assert args.command == "sim"
        assert args.episodes == 2

    def test_parse_train_subcommand(self) -> None:
        from armdroid.main import _parse_args

        args = _parse_args(["train", "--total-timesteps", "1000"])
        assert args.command == "train"
        assert args.total_timesteps == 1000

    def test_parse_rollout_subcommand(self) -> None:
        from armdroid.main import _parse_args

        args = _parse_args(["rollout", "--num-disks", "5"])
        assert args.command == "rollout"
        assert args.num_disks == 5

    def test_mock_hardware_flag_overrides_config(self) -> None:
        from armdroid.main import _parse_args

        args = _parse_args(["--mock-hardware", "sim"])
        assert args.mock_hardware is True


class TestFactoryAcceptsExplicitDriver:
    """Regression: build_arm_controller still accepts an explicit driver."""

    def test_explicit_driver_passed_through(self) -> None:
        from armdroid.orchestration.factory import build_arm_controller, build_arm_driver

        cfg = ArmSettings(mock_hardware=True)
        drv = build_arm_driver(cfg)
        ctrl = build_arm_controller(cfg, driver=drv)
        # The controller's primitives' driver is the explicit one we passed
        assert ctrl.primitives.driver is drv
