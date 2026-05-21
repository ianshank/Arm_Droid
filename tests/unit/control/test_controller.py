"""Tests for ArmController wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from armdroid.config.schema import ArmConfig, ArmTrainingConfig
from armdroid.control.controller import ArmController
from armdroid.control.primitives import ActionPrimitives
from armdroid.control.sac_agent import SACAgent
from armdroid.hardware.mock_arm_driver import MockArmDriver


def _make_controller() -> ArmController:
    arm_cfg = ArmConfig(dof=6, home_position=[0.0] * 6)
    driver = MockArmDriver(arm_cfg)
    agent = SACAgent(ArmTrainingConfig())
    primitives = ActionPrimitives(arm_cfg, driver)
    return ArmController(agent, primitives)


class TestExecuteAction:
    """Tests for execute_action."""

    @pytest.mark.asyncio
    async def test_success_returns_true(self) -> None:
        ctrl = _make_controller()
        action = np.zeros(6, dtype=np.float64)
        result = await ctrl.execute_action(action)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_wrong_dof_returns_not_success(self) -> None:
        ctrl = _make_controller()
        bad_action = np.zeros(2, dtype=np.float64)  # wrong DOF
        result = await ctrl.execute_action(bad_action)
        # primitives.transit catches the ValueError and returns False
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_unexpected_exception_returns_failure_dict(self) -> None:
        from unittest.mock import AsyncMock, patch

        ctrl = _make_controller()
        action = np.zeros(6, dtype=np.float64)
        # Force an unexpected exception that bypasses the primitives try/except
        with patch.object(ctrl._primitives, "transit", AsyncMock(side_effect=RuntimeError("boom"))):
            result = await ctrl.execute_action(action)
        assert result["success"] is False
        assert "error" in result["info"]


class TestBuildForEnv:
    """build_for_env wires the SAC agent to an environment."""

    def test_build_for_env_builds_agent(self) -> None:
        from unittest.mock import MagicMock, patch

        from armdroid.domain.protocols import ArmEnvironmentProtocol

        ctrl = _make_controller()
        assert not ctrl.agent.is_built

        # spec= pins the mock to the single-env protocol; without it
        # bare MagicMock ambiguously satisfies BOTH single-env AND
        # VecArmEnvironmentProtocol (every attribute auto-exists), which
        # would trip the vec dispatch path against a non-vec SAC agent.
        mock_env = MagicMock(spec=ArmEnvironmentProtocol)
        with (
            patch("armdroid.control.sac_agent.SAC") as mock_sac_cls,
            patch("armdroid.control.sac_agent.HerReplayBuffer"),
        ):
            mock_sac_cls.return_value = MagicMock()
            ctrl.build_for_env(mock_env)

        assert ctrl.agent.is_built

    def test_build_for_env_idempotent(self) -> None:
        from unittest.mock import MagicMock, patch

        from armdroid.domain.protocols import ArmEnvironmentProtocol

        ctrl = _make_controller()
        mock_env = MagicMock(spec=ArmEnvironmentProtocol)
        with (
            patch("armdroid.control.sac_agent.SAC") as mock_sac_cls,
            patch("armdroid.control.sac_agent.HerReplayBuffer"),
        ):
            mock_sac_cls.return_value = MagicMock()
            ctrl.build_for_env(mock_env)
            ctrl.build_for_env(mock_env)  # second call should not re-build
            assert mock_sac_cls.call_count == 1


class TestTrainPolicy:
    """train_policy delegates to the agent's train+save."""

    def test_train_policy_returns_path(self) -> None:
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from armdroid.domain.protocols import ArmEnvironmentProtocol

        ctrl = _make_controller()
        mock_env = MagicMock(spec=ArmEnvironmentProtocol)
        with (
            patch("armdroid.control.sac_agent.SAC") as mock_sac_cls,
            patch("armdroid.control.sac_agent.HerReplayBuffer"),
        ):
            mock_model = MagicMock()
            mock_sac_cls.return_value = mock_model
            mock_model.save = MagicMock()
            ctrl.build_for_env(mock_env)
            path = ctrl.train_policy(total_timesteps=10)

        assert isinstance(path, Path)
        mock_model.learn.assert_called_once_with(total_timesteps=10)


class TestExecutePrimitive:
    """Tests for execute_primitive."""

    @pytest.mark.asyncio
    async def test_grasp(self) -> None:
        ctrl = _make_controller()
        target = np.zeros(6, dtype=np.float64)
        result = await ctrl.execute_primitive("grasp", target)
        assert result is True

    @pytest.mark.asyncio
    async def test_place(self) -> None:
        ctrl = _make_controller()
        target = np.zeros(6, dtype=np.float64)
        result = await ctrl.execute_primitive("place", target)
        assert result is True

    @pytest.mark.asyncio
    async def test_transit(self) -> None:
        ctrl = _make_controller()
        target = np.zeros(6, dtype=np.float64)
        result = await ctrl.execute_primitive("transit", target)
        assert result is True

    @pytest.mark.asyncio
    async def test_home(self) -> None:
        ctrl = _make_controller()
        target = np.zeros(6, dtype=np.float64)
        result = await ctrl.execute_primitive("home", target)
        assert result is True

    @pytest.mark.asyncio
    async def test_unknown_primitive_returns_false(self) -> None:
        ctrl = _make_controller()
        target = np.zeros(6, dtype=np.float64)
        result = await ctrl.execute_primitive("nonexistent", target)
        assert result is False
