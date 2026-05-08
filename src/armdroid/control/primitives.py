"""Pre-trained grasp / place / transit / home primitives for the arm.

Modular action library that abstracts away low-level joint trajectory
planning. Each primitive accepts a target pose (NDArray) and dispatches
to the underlying :class:`ArmDriverProtocol`.

The primitives speak the legacy driver surface
(``send_joint_command`` / ``open_gripper`` / ``close_gripper``) by
default for backwards compatibility with controllers that rely on the
historical "set-and-stick" semantics. When the configured arm includes
an explicit gripper joint (``cfg.gripper_joint_index is not None``) the
gripper is folded into the joint vector and primitives skip the legacy
gripper calls.

Per-primitive interpolation durations come from
:class:`armdroid.config.schema.ArmConfig` (``transit_duration_s``,
``grasp_duration_s``, ``place_duration_s``, ``home_duration_s``) so no
durations are hardcoded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from armdroid.logging.setup import get_logger

if TYPE_CHECKING:
    from armdroid.config.schema import ArmConfig
    from armdroid.protocols import ArmDriverProtocol

_log = get_logger(__name__)


class ActionPrimitives:
    """Library of pre-trained manipulation primitives.

    Args:
        arm_cfg: Arm hardware configuration (dof, home_position,
            transit/grasp/place durations, gripper_joint_index).
        driver: Driver implementing ``ArmDriverProtocol``.
    """

    def __init__(self, arm_cfg: ArmConfig, driver: ArmDriverProtocol) -> None:
        """Initialise primitives.

        Args:
            arm_cfg: Arm config with joint limits, home position, durations.
            driver: Arm driver for commanding joints.
        """
        self._cfg = arm_cfg
        self._driver = driver
        self._home = np.array(arm_cfg.home_position, dtype=np.float64)
        _log.info(
            "action_primitives_init",
            dof=arm_cfg.dof,
            gripper_joint_index=arm_cfg.gripper_joint_index,
        )

    @property
    def driver(self) -> ArmDriverProtocol:
        """Return the arm driver (for orchestrator handoff)."""
        return self._driver

    async def transit(self, target_pose: NDArray[np.float64]) -> bool:
        """Move arm to target joint pose without grasping.

        Args:
            target_pose: Target joint angles, shape ``(dof,)``.

        Returns:
            True if transit completed successfully.
        """
        _log.debug("primitive_transit", target=target_pose.tolist())
        try:
            await self._driver.send_joint_command(target_pose)
        except Exception:
            _log.error("transit_failed", exc_info=True)
            return False
        return True

    async def grasp(self, pre_grasp_pose: NDArray[np.float64]) -> float:
        """Execute grasp primitive — approach pose, then close gripper.

        When ``cfg.gripper_joint_index is not None`` the gripper close is
        baked into the joint vector by writing ``1.0`` to the gripper
        joint via the modern ``send_joint_positions`` API. Otherwise the
        legacy ``close_gripper`` driver method is used.

        Args:
            pre_grasp_pose: Joint angles for pre-grasp position.

        Returns:
            Measured grip force (``0.0`` if grasp failed).
        """
        _log.debug("primitive_grasp", pre_grasp=pre_grasp_pose.tolist())
        try:
            await self._driver.send_joint_command(pre_grasp_pose)
        except Exception:
            _log.error("grasp_approach_failed", exc_info=True)
            return 0.0

        if self._cfg.gripper_joint_index is not None:
            # Gripper is an explicit protocol joint — close by writing the
            # joint vector with the gripper at 1.0 (closed). The other
            # joints are held at their just-commanded positions.
            try:
                grip_joints = list(pre_grasp_pose.astype(np.float64))
                grip_joints[self._cfg.gripper_joint_index] = 1.0
                await self._driver.send_joint_positions(
                    tuple(grip_joints),
                    duration_s=self._cfg.grasp_duration_s,
                )
            except Exception:
                _log.error("grasp_close_failed", exc_info=True)
                return 0.0
            _log.info("grasp_complete", force=1.0, gripper_joint_close=True)
            return 1.0

        # Legacy gripper path
        try:
            force = await self._driver.close_gripper()
        except Exception:
            _log.error("grasp_close_failed", exc_info=True)
            return 0.0
        _log.info("grasp_complete", force=force)
        return force

    async def place(self, place_pose: NDArray[np.float64]) -> bool:
        """Execute place primitive — move to target pose, then open gripper.

        When ``cfg.gripper_joint_index is not None`` the gripper open is
        baked into the joint vector (gripper joint = ``0.0``) instead of
        calling the legacy ``open_gripper`` driver method.

        Args:
            place_pose: Joint angles for place position.

        Returns:
            True if place completed successfully.
        """
        _log.debug("primitive_place", target=place_pose.tolist())
        try:
            await self._driver.send_joint_command(place_pose)
        except Exception:
            _log.error("place_approach_failed", exc_info=True)
            return False

        if self._cfg.gripper_joint_index is not None:
            try:
                release_joints = list(place_pose.astype(np.float64))
                release_joints[self._cfg.gripper_joint_index] = 0.0
                await self._driver.send_joint_positions(
                    tuple(release_joints),
                    duration_s=self._cfg.place_duration_s,
                )
            except Exception:
                _log.error("place_open_failed", exc_info=True)
                return False
            _log.info("place_complete", gripper_joint_open=True)
            return True

        # Legacy gripper path
        try:
            await self._driver.open_gripper()
        except Exception:
            _log.error("place_open_failed", exc_info=True)
            return False
        _log.info("place_complete")
        return True

    async def home(self) -> bool:
        """Return arm to home position.

        Returns:
            True if home completed successfully.
        """
        _log.debug("primitive_home")
        try:
            await self._driver.home()
        except Exception:
            _log.error("home_failed", exc_info=True)
            return False
        return True
