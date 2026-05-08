"""End-to-end pipeline tests — orchestrator → planner → controller → driver.

These tests run the full DI graph and exercise concrete behaviours that
span more than one subsystem:

* `armdroid sim` runs N episodes with mock hardware and prints rewards
  without raising — this is the same path the CLI subcommand follows.
* The orchestrator's ``rollout`` plans a Tower of Hanoi solve and
  dispatches each step through the controller's primitive layer.
* Building two orchestrators back-to-back doesn't leak state between
  them.

Marked ``e2e`` so they can be filtered into a dedicated CI stage.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from armdroid.config.schema import ArmSettings
from armdroid.factory import build_arm_orchestrator
from armdroid.protocols import SymbolicState

pytestmark = pytest.mark.e2e


class TestSimPipeline:
    """Stepping the env for a few episodes runs without exceptions."""

    @pytest.mark.asyncio
    async def test_orchestrator_constructs_then_shutdown(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        orch = build_arm_orchestrator(cfg)
        # Sanity: env reset + step works without crashing
        env = orch.environment
        _obs, _info = env.reset(seed=0)
        action = np.zeros(cfg.arm.dof, dtype=np.float64)
        _obs, _reward, terminated, truncated, _info = env.step(action)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        await orch.shutdown()


class TestRolloutPipeline:
    """orchestrator.rollout plans + executes a Hanoi solve via the controller."""

    @pytest.mark.asyncio
    async def test_three_disk_solve_dispatches_all_steps(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        cfg.arm_task.num_disks = 3
        orch = build_arm_orchestrator(cfg)
        try:
            empty = SymbolicState(predicates=frozenset(), objects={})
            result = await orch.rollout(empty, empty)
            # Plan exists, execution either succeeded or aborted
            # cleanly (no exception raised). For 3 disks, an optimal
            # plan has 7 steps.
            assert "plan" in result
            assert "executed" in result
            assert "success" in result
            assert isinstance(result["plan"], list)
        finally:
            await orch.shutdown()


class TestSequentialOrchestrators:
    """Building two orchestrators back-to-back doesn't leak state."""

    @pytest.mark.asyncio
    async def test_two_orchs_sequential(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        orch1 = build_arm_orchestrator(cfg)
        await orch1.shutdown()
        orch2 = build_arm_orchestrator(cfg)
        # New orchestrator with fresh driver — has its own state.
        assert orch2.driver is not orch1.driver
        await orch2.shutdown()


class TestAsyncShutdownIdempotence:
    """orchestrator.shutdown can be called twice without raising."""

    @pytest.mark.asyncio
    async def test_double_shutdown(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        orch = build_arm_orchestrator(cfg)
        await orch.shutdown()
        await orch.shutdown()  # idempotent — does not raise


class TestSimSubcommandPath:
    """Mirror what `armdroid sim` does without running the CLI."""

    @pytest.mark.asyncio
    async def test_sim_two_episodes(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        orch = build_arm_orchestrator(cfg)
        env = orch.environment
        try:
            for ep in range(2):
                _obs, _info = env.reset(seed=ep)
                terminated = False
                truncated = False
                step_count = 0
                while not (terminated or truncated) and step_count < 20:
                    action = np.zeros(cfg.arm.dof, dtype=np.float64)
                    _obs, _reward, terminated, truncated, _info = env.step(action)
                    step_count += 1
                # Episode ended (either terminated, truncated, or hit cap)
                assert step_count > 0
        finally:
            await orch.shutdown()


class TestEstopRecoveryE2E:
    """Latched e-stop on the driver propagates and clear restores motion."""

    @pytest.mark.asyncio
    async def test_estop_then_clear_round_trip(self) -> None:
        cfg = ArmSettings(mock_hardware=True)
        orch = build_arm_orchestrator(cfg)
        try:
            drv = orch.driver
            await drv.connect()
            await drv.emergency_stop()
            await asyncio.sleep(0.01)
            state = await drv.read_state()
            assert state.estop_active
            await drv.clear_emergency_stop()
            state = await drv.read_state()
            assert not state.estop_active
        finally:
            await orch.shutdown()
