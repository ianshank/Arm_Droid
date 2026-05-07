"""Tests for armdroid.orchestrator.ArmOrchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from armdroid.config.schema import ArmSettings
from armdroid.factory import build_arm_orchestrator
from armdroid.orchestrator import ArmOrchestrator


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
        agent = orchestrator.controller.agent  # type: ignore[attr-defined]
        assert not agent.is_built

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
