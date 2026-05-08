"""Primitives tests covering the gripper-as-joint code path (dof >= 7).

The existing ``test_primitives.py`` covers the legacy 6-DoF path where
the gripper is controlled via the driver's ``open_gripper`` /
``close_gripper`` methods. This module exercises the modern path where
``ArmConfig.gripper_joint_index`` is not None and the primitives fold
the gripper into the joint vector via ``send_joint_positions``.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pytest

from armdroid.config.schema import (
    ArmConfig,
    ArmServoConfig,
    JointLimits,
)
from armdroid.control.primitives import ActionPrimitives
from armdroid.hardware.mock_arm_driver import MockArmDriver

_GENEROUS_LIMITS: Final = JointLimits(
    min_rad=-math.pi,
    max_rad=math.pi,
    max_velocity_rad_s=10.0,
)
_GRIPPER_LIMITS: Final = JointLimits(
    min_rad=0.0,
    max_rad=1.0,
    max_velocity_rad_s=5.0,
)


def _make_7dof_cfg() -> ArmConfig:
    return ArmConfig(
        dof=7,
        home_position=[0.0] * 7,
        joint_limits=[_GENEROUS_LIMITS] * 6 + [_GRIPPER_LIMITS],
        servos=[ArmServoConfig(pwm_pin=13 + i) for i in range(7)],
    )


def _make_primitives_7dof() -> tuple[ActionPrimitives, MockArmDriver]:
    cfg = _make_7dof_cfg()
    driver = MockArmDriver(cfg)
    return ActionPrimitives(cfg, driver), driver


class TestGripperJointIndex:
    """`ArmConfig.gripper_joint_index` is None at dof < 7, dof - 1 otherwise."""

    def test_returns_none_at_dof_six(self) -> None:
        cfg = ArmConfig()
        assert cfg.gripper_joint_index is None

    def test_returns_last_index_at_dof_seven(self) -> None:
        cfg = _make_7dof_cfg()
        assert cfg.gripper_joint_index == 6

    def test_returns_last_index_at_dof_eight(self) -> None:
        cfg = ArmConfig(dof=8)
        assert cfg.gripper_joint_index == 7


class TestGraspWithJointGripper:
    """Grasp on a 7-DoF arm writes joint dof-1 to 1.0."""

    @pytest.mark.asyncio
    async def test_grasp_sets_gripper_joint_to_one(self) -> None:
        prims, driver = _make_primitives_7dof()
        pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        force = await prims.grasp(pose)
        assert force == 1.0
        # The gripper joint should now be 1.0 in the driver's view.
        # Read state via legacy adapter (auto-connects).
        joints = await driver.get_joint_states()
        # The mock interpolates over grasp_duration_s; consult internal
        # state directly to verify the segment_target.
        assert driver._segment_target[6] == pytest.approx(1.0)
        # Joints 0..5 should still be the pose values (modern API uses
        # `send_joint_positions` for both transit and gripper close, so
        # all joints get written).
        np.testing.assert_allclose(joints[:6], pose[:6], atol=1e-6)

    @pytest.mark.asyncio
    async def test_grasp_uses_grasp_duration_s(self) -> None:
        cfg = _make_7dof_cfg()
        cfg = cfg.model_copy(update={"grasp_duration_s": 0.75})
        driver = MockArmDriver(cfg)
        prims = ActionPrimitives(cfg, driver)
        pose = np.zeros(7, dtype=np.float64)
        await prims.grasp(pose)
        assert driver._segment_duration_s == pytest.approx(0.75)


class TestPlaceWithJointGripper:
    """Place on a 7-DoF arm writes joint dof-1 to 0.0."""

    @pytest.mark.asyncio
    async def test_place_sets_gripper_joint_to_zero(self) -> None:
        prims, driver = _make_primitives_7dof()
        # Pre-close the gripper so place's open is observable
        gripper_closed = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        await prims.grasp(gripper_closed)
        assert driver._segment_target[6] == pytest.approx(1.0)
        # Now place at the same pose; the place primitive overrides
        # the gripper joint to 0.0 (open).
        result = await prims.place(gripper_closed)
        assert result is True
        assert driver._segment_target[6] == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_place_uses_place_duration_s(self) -> None:
        cfg = _make_7dof_cfg()
        cfg = cfg.model_copy(update={"place_duration_s": 0.5})
        driver = MockArmDriver(cfg)
        prims = ActionPrimitives(cfg, driver)
        pose = np.zeros(7, dtype=np.float64)
        await prims.place(pose)
        assert driver._segment_duration_s == pytest.approx(0.5)


class TestPerPrimitiveDurations:
    """Per-primitive duration fields land in the config and can be overridden."""

    def test_default_durations(self) -> None:
        cfg = ArmConfig()
        assert cfg.transit_duration_s == 2.0
        assert cfg.grasp_duration_s == 1.0
        assert cfg.place_duration_s == 1.0
        assert cfg.home_duration_s == 2.0

    def test_durations_must_be_positive(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ArmConfig(grasp_duration_s=0.0)
        with pytest.raises(ValidationError):
            ArmConfig(place_duration_s=-1.0)
