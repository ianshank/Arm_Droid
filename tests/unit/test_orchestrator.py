"""Tests for armdroid.orchestrator.ArmOrchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from armdroid.config.schema import ArmSettings
from armdroid.control.controller import ArmController
from armdroid.domain.state import PlanStep, SymbolicState
from armdroid.orchestration.factory import build_arm_orchestrator
from armdroid.orchestration.orchestrator import (
    ArmOrchestrator,
    _resolve_target_position,
    _step_args_to_target,
)


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
    """_step_args_to_target adapter (legacy shim, always zero)."""

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


class TestResolveTargetPosition:
    """_resolve_target_position resolves PDDL args via task_cfg (TD-5)."""

    def test_falls_back_to_zeros_without_task_cfg(self) -> None:
        import numpy as np

        target, used_fallback = _resolve_target_position(
            ["disk_1", "peg_a", "peg_c"], task_cfg=None
        )
        np.testing.assert_array_equal(target, np.zeros(3, dtype=np.float64))
        assert used_fallback is True

    def test_resolves_destination_peg_from_task_cfg(self) -> None:
        import numpy as np

        from armdroid.config.schema import ArmTaskConfig

        cfg = ArmTaskConfig(
            num_pegs=3,
            peg_positions=[
                [0.21, 0.0, 0.0],
                [0.31, 0.0, 0.0],
                [0.41, 0.0, 0.0],
            ],
        )
        target, used_fallback = _resolve_target_position(["disk_1", "peg_a", "peg_c"], task_cfg=cfg)
        np.testing.assert_allclose(target, [0.41, 0.0, 0.0])
        assert used_fallback is False

    def test_resolves_first_peg_when_only_arg(self) -> None:
        from armdroid.config.schema import ArmTaskConfig

        cfg = ArmTaskConfig()  # default 3 pegs
        target, used_fallback = _resolve_target_position(["peg_a"], task_cfg=cfg)
        assert tuple(target) == tuple(cfg.peg_positions[0])
        assert used_fallback is False

    def test_resolves_basket_for_laundry_args(self) -> None:
        from armdroid.config.schema import ArmTaskConfig

        cfg = ArmTaskConfig(
            num_baskets=3,
            basket_positions=[
                [0.5, -0.2, 0.0],
                [0.6, -0.2, 0.0],
                [0.7, -0.2, 0.0],
            ],
        )
        target, used_fallback = _resolve_target_position(["shirt_1", "basket_b"], task_cfg=cfg)
        assert tuple(target) == tuple(cfg.basket_positions[1])
        assert used_fallback is False

    def test_unrecognised_name_falls_back(self) -> None:
        import numpy as np

        from armdroid.config.schema import ArmTaskConfig

        cfg = ArmTaskConfig()
        target, used_fallback = _resolve_target_position(
            ["mystery_object", "unknown_zone"], task_cfg=cfg
        )
        np.testing.assert_array_equal(target, np.zeros(3, dtype=np.float64))
        assert used_fallback is True

    def test_out_of_range_index_falls_back(self) -> None:
        import numpy as np

        from armdroid.config.schema import ArmTaskConfig

        cfg = ArmTaskConfig()  # 3 pegs by default; peg_z = index 25
        target, used_fallback = _resolve_target_position(["peg_z"], task_cfg=cfg)
        np.testing.assert_array_equal(target, np.zeros(3, dtype=np.float64))
        assert used_fallback is True

    def test_case_insensitive_name_match(self) -> None:
        from armdroid.config.schema import ArmTaskConfig

        cfg = ArmTaskConfig()
        # Planner emits lowercase but case-insensitive match keeps
        # legacy callers passing "peg_C" working.
        target_lc, used_lc = _resolve_target_position(["peg_c"], task_cfg=cfg)
        target_uc, used_uc = _resolve_target_position(["peg_C"], task_cfg=cfg)
        assert used_lc is False
        assert used_uc is False
        assert tuple(target_lc) == tuple(target_uc)

    def test_destination_is_last_arg(self) -> None:
        from armdroid.config.schema import ArmTaskConfig

        cfg = ArmTaskConfig(
            num_pegs=3,
            peg_positions=[
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0],
            ],
        )
        # Hanoi convention: ["disk", source, destination] - destination
        # is last, so peg_a (source) must NOT be returned.
        target, _ = _resolve_target_position(["disk_1", "peg_a", "peg_b"], task_cfg=cfg)
        assert tuple(target) == tuple(cfg.peg_positions[1])

    def test_returns_dtype_float64(self) -> None:
        import numpy as np

        from armdroid.config.schema import ArmTaskConfig

        cfg = ArmTaskConfig()
        target, _ = _resolve_target_position(["peg_a"], task_cfg=cfg)
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
