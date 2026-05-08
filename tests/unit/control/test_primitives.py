"""Tests for arm action primitives."""

from __future__ import annotations

import numpy as np
import pytest

from armdroid.config.schema import ArmConfig
from armdroid.control.primitives import ActionPrimitives
from armdroid.hardware.mock_arm_driver import MockArmDriver


def _make_cfg() -> ArmConfig:
    return ArmConfig(dof=6, home_position=[0.0] * 6)


def _make_primitives() -> tuple[ActionPrimitives, MockArmDriver]:
    cfg = _make_cfg()
    driver = MockArmDriver(cfg)
    return ActionPrimitives(cfg, driver), driver


class TestTransit:
    """Tests for transit primitive."""

    @pytest.mark.asyncio
    async def test_transit_success(self) -> None:
        prims, _driver = _make_primitives()
        target = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
        result = await prims.transit(target)
        assert result is True

    @pytest.mark.asyncio
    async def test_transit_updates_joints(self) -> None:
        prims, driver = _make_primitives()
        target = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
        await prims.transit(target)
        joints = await driver.get_joint_states()
        np.testing.assert_allclose(joints, target)

    @pytest.mark.asyncio
    async def test_transit_failure_returns_false(self) -> None:
        prims, _driver = _make_primitives()
        # Wrong DOF to trigger ValueError
        target = np.array([0.1, 0.2], dtype=np.float64)
        result = await prims.transit(target)
        assert result is False


class TestGrasp:
    """Tests for grasp primitive."""

    @pytest.mark.asyncio
    async def test_grasp_returns_force(self) -> None:
        prims, _driver = _make_primitives()
        pose = np.zeros(6, dtype=np.float64)
        force = await prims.grasp(pose)
        assert force > 0.0

    @pytest.mark.asyncio
    async def test_grasp_failure_returns_zero(self) -> None:
        prims, _driver = _make_primitives()
        bad_pose = np.array([0.1], dtype=np.float64)
        force = await prims.grasp(bad_pose)
        assert force == 0.0


class TestPlace:
    """Tests for place primitive."""

    @pytest.mark.asyncio
    async def test_place_success(self) -> None:
        prims, _driver = _make_primitives()
        pose = np.zeros(6, dtype=np.float64)
        result = await prims.place(pose)
        assert result is True

    @pytest.mark.asyncio
    async def test_place_failure_returns_false(self) -> None:
        prims, _driver = _make_primitives()
        bad_pose = np.array([0.1], dtype=np.float64)
        result = await prims.place(bad_pose)
        assert result is False


class TestHome:
    """Tests for home primitive."""

    @pytest.mark.asyncio
    async def test_home_success(self) -> None:
        prims, _driver = _make_primitives()
        result = await prims.home()
        assert result is True

    @pytest.mark.asyncio
    async def test_home_exception_returns_false(self) -> None:
        from unittest.mock import AsyncMock, patch

        prims, _driver = _make_primitives()
        with patch.object(_driver, "home", AsyncMock(side_effect=RuntimeError("driver error"))):
            result = await prims.home()
        assert result is False


# ---------------------------------------------------------------------------
# Modern path (dof >= 7, gripper_joint_index is not None)
# ---------------------------------------------------------------------------


def _make_modern_primitives() -> tuple[ActionPrimitives, MockArmDriver]:
    """Return primitives backed by a 7-DoF driver (modern gripper path)."""
    cfg = ArmConfig(dof=7, home_position=[0.0] * 7)
    driver = MockArmDriver(cfg)
    return ActionPrimitives(cfg, driver), driver


class TestGraspModernPath:
    """Modern-path grasp tests (dof=7, gripper_joint_index=6)."""

    @pytest.mark.asyncio
    async def test_grasp_modern_success_returns_force(self) -> None:
        from unittest.mock import AsyncMock, patch

        prims, driver = _make_modern_primitives()
        pose = np.zeros(7, dtype=np.float64)
        with patch.object(driver, "send_joint_positions", new_callable=AsyncMock):
            force = await prims.grasp(pose)
        assert force == 1.0

    @pytest.mark.asyncio
    async def test_grasp_modern_approach_failure_returns_zero(self) -> None:
        """Lines 121-123: exception on approach send_joint_positions."""
        from unittest.mock import AsyncMock, patch

        prims, driver = _make_modern_primitives()
        pose = np.zeros(7, dtype=np.float64)
        with patch.object(
            driver,
            "send_joint_positions",
            AsyncMock(side_effect=RuntimeError("approach fail")),
        ):
            force = await prims.grasp(pose)
        assert force == 0.0

    @pytest.mark.asyncio
    async def test_grasp_modern_close_failure_returns_zero(self) -> None:
        """Lines 132-134: exception on gripper-close send_joint_positions."""
        from unittest.mock import patch

        prims, driver = _make_modern_primitives()
        pose = np.zeros(7, dtype=np.float64)
        call_count = {"n": 0}

        async def _approach_ok_then_fail(*_args: object, **_kwargs: object) -> None:
            call_count["n"] += 1
            if call_count["n"] > 1:
                raise RuntimeError("gripper close fail")

        with patch.object(driver, "send_joint_positions", side_effect=_approach_ok_then_fail):
            force = await prims.grasp(pose)
        assert force == 0.0


class TestGraspLegacyCloseFailure:
    """Line 146-148: legacy close_gripper() raises."""

    @pytest.mark.asyncio
    async def test_grasp_legacy_close_failure_returns_zero(self) -> None:
        from unittest.mock import AsyncMock, patch

        prims, driver = _make_primitives()
        pose = np.zeros(6, dtype=np.float64)
        with patch.object(
            driver,
            "close_gripper",
            AsyncMock(side_effect=RuntimeError("gripper jam")),
        ):
            force = await prims.grasp(pose)
        assert force == 0.0


class TestPlaceModernPath:
    """Modern-path place tests (dof=7, gripper_joint_index=6)."""

    @pytest.mark.asyncio
    async def test_place_modern_success_returns_true(self) -> None:
        from unittest.mock import AsyncMock, patch

        prims, driver = _make_modern_primitives()
        pose = np.zeros(7, dtype=np.float64)
        with patch.object(driver, "send_joint_positions", new_callable=AsyncMock):
            result = await prims.place(pose)
        assert result is True

    @pytest.mark.asyncio
    async def test_place_modern_approach_failure_returns_false(self) -> None:
        """Lines 175-177: exception on approach send_joint_positions."""
        from unittest.mock import AsyncMock, patch

        prims, driver = _make_modern_primitives()
        pose = np.zeros(7, dtype=np.float64)
        with patch.object(
            driver,
            "send_joint_positions",
            AsyncMock(side_effect=RuntimeError("approach fail")),
        ):
            result = await prims.place(pose)
        assert result is False

    @pytest.mark.asyncio
    async def test_place_modern_release_failure_returns_false(self) -> None:
        """Lines 186-188: exception on gripper-open send_joint_positions."""
        from unittest.mock import patch

        prims, driver = _make_modern_primitives()
        pose = np.zeros(7, dtype=np.float64)
        call_count = {"n": 0}

        async def _approach_ok_then_fail(*_args: object, **_kwargs: object) -> None:
            call_count["n"] += 1
            if call_count["n"] > 1:
                raise RuntimeError("gripper open fail")

        with patch.object(driver, "send_joint_positions", side_effect=_approach_ok_then_fail):
            result = await prims.place(pose)
        assert result is False


class TestPlaceLegacyOpenFailure:
    """Line 200-202: legacy open_gripper() raises."""

    @pytest.mark.asyncio
    async def test_place_legacy_open_failure_returns_false(self) -> None:
        from unittest.mock import AsyncMock, patch

        prims, driver = _make_primitives()
        pose = np.zeros(6, dtype=np.float64)
        with patch.object(
            driver,
            "open_gripper",
            AsyncMock(side_effect=RuntimeError("gripper release fail")),
        ):
            result = await prims.place(pose)
        assert result is False
