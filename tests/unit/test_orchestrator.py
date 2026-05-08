"""Tests for armdroid.orchestrator.ArmOrchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from armdroid.config.schema import ArmSettings
from armdroid.control.controller import ArmController
from armdroid.domain.state import PlanStep, SymbolicState
from armdroid.orchestration.factory import build_arm_orchestrator
from armdroid.orchestration.orchestrator import ArmOrchestrator, _step_args_to_target


@pytest.fixture
def orchestrator() -> ArmOrchestrator:
    """Build a fully wired orchestrator with mock hardware."""
    return build_arm_orchestrator(ArmSettings(mock_hardware=True))


class TestArmOrchestratorWiring:
    """The orchestrator exposes the five subsystems as properties."""

    def test_properties_match_constructor_args(self, orchestrator: ArmOrchestrator) -> None:
        assert orchestrator.perception is not None
        assert orchestrator.planner is not None
        assert orchestrator.controller is not None
        assert orchestrator.environment is not None
        assert orchestrator.driver is not None


class TestArmOrchestratorTrain:
    """orchestrator.train wires the SAC agent to the env on first call."""

    def test_train_builds_agent_lazily(self, orchestrator: ArmOrchestrator) -> None:
        """The agent must be unbuilt before train() and built after."""
        ctrl = orchestrator.controller
        assert isinstance(ctrl, ArmController)
        assert not ctrl.agent.is_built

        with (
            patch("armdroid.control.sac_agent.SAC") as mock_sac_cls,
            patch("armdroid.control.sac_agent.HerReplayBuffer"),
        ):
            mock_model = MagicMock()
            mock_sac_cls.return_value = mock_model
            mock_model.save = MagicMock()

            orchestrator.train(total_timesteps=100)

            mock_sac_cls.assert_called_once()
            mock_model.learn.assert_called_once_with(total_timesteps=100)
            mock_model.save.assert_called_once()

    def test_train_skips_build_if_already_built(self, orchestrator: ArmOrchestrator) -> None:
        """build_for_env should not rebuild a model that is already built."""
        ctrl = orchestrator.controller
        assert isinstance(ctrl, ArmController)

        with (
            patch("armdroid.control.sac_agent.SAC") as mock_sac_cls,
            patch("armdroid.control.sac_agent.HerReplayBuffer"),
        ):
            mock_model = MagicMock()
            mock_sac_cls.return_value = mock_model
            mock_model.save = MagicMock()

            orchestrator.train(total_timesteps=50)
            orchestrator.train(total_timesteps=50)

            # SAC constructor called only once (second train reuses the built model)
            assert mock_sac_cls.call_count == 1


class TestArmOrchestratorRollout:
    """orchestrator.rollout plans and dispatches steps through the controller."""

    @pytest.mark.asyncio
    async def test_rollout_success_all_steps_executed(self, orchestrator: ArmOrchestrator) -> None:
        initial = SymbolicState(predicates=frozenset(), objects={})
        goal = SymbolicState(predicates=frozenset(), objects={})
        plan_stub = [PlanStep("move", ["disk1", "peg_A", "peg_C"])]

        mock_exec = AsyncMock(return_value=True)
        with (
            patch.object(orchestrator._planner, "plan", return_value=plan_stub),
            patch.object(orchestrator._controller, "execute_primitive", mock_exec),
        ):
            result = await orchestrator.rollout(initial, goal)

        assert result["success"] is True
        assert result["executed"] == 1
        assert result["plan"] is plan_stub

    @pytest.mark.asyncio
    async def test_rollout_aborts_on_step_failure(self, orchestrator: ArmOrchestrator) -> None:
        initial = SymbolicState(predicates=frozenset(), objects={})
        goal = SymbolicState(predicates=frozenset(), objects={})
        plan_stub = [
            PlanStep("move", ["disk1", "peg_A", "peg_B"]),
            PlanStep("move", ["disk2", "peg_A", "peg_C"]),
        ]

        with (
            patch.object(orchestrator._planner, "plan", return_value=plan_stub),
            patch.object(
                orchestrator._controller,
                "execute_primitive",
                AsyncMock(side_effect=[False, True]),
            ),
        ):
            result = await orchestrator.rollout(initial, goal)

        assert result["success"] is False
        assert result["executed"] == 0

    @pytest.mark.asyncio
    async def test_rollout_empty_plan(self, orchestrator: ArmOrchestrator) -> None:
        initial = SymbolicState(predicates=frozenset(), objects={})
        goal = SymbolicState(predicates=frozenset(), objects={})

        with patch.object(orchestrator._planner, "plan", return_value=[]):
            result = await orchestrator.rollout(initial, goal)

        assert result["success"] is True
        assert result["executed"] == 0


class TestStepArgsToTarget:
    """_step_args_to_target adapter."""

    def test_returns_zero_array_with_args(self) -> None:
        import numpy as np

        target = _step_args_to_target(["disk1", "peg_A", "peg_C"])
        np.testing.assert_array_equal(target, np.zeros(3, dtype=np.float64))

    def test_returns_zero_array_without_args(self) -> None:
        import numpy as np

        target = _step_args_to_target([])
        np.testing.assert_array_equal(target, np.zeros(3, dtype=np.float64))

    def test_returns_float64(self) -> None:
        import numpy as np

        target = _step_args_to_target(["x"])
        assert target.dtype == np.float64


class TestArmOrchestratorShutdown:
    """orchestrator.shutdown stops the driver and closes the env."""

    @pytest.mark.asyncio
    async def test_shutdown_calls_driver_stop(self, orchestrator: ArmOrchestrator) -> None:
        with (
            patch.object(orchestrator._driver, "stop", AsyncMock()) as mock_stop,
            patch.object(orchestrator._environment, "close") as mock_close,
        ):
            await orchestrator.shutdown()
            mock_stop.assert_awaited_once()
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_swallows_exceptions(self, orchestrator: ArmOrchestrator) -> None:
        with (
            patch.object(
                orchestrator._driver,
                "stop",
                AsyncMock(side_effect=RuntimeError("driver dead")),
            ),
            patch.object(
                orchestrator._environment,
                "close",
                MagicMock(side_effect=RuntimeError("env dead")),
            ),
        ):
            # Should not raise — failures during shutdown are logged but swallowed.
            await orchestrator.shutdown()
