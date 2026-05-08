"""Tests for adaptive replanner.

These are *unit* tests for the Replanner class itself.  The SymbolicPlanner
dependency is mocked so the tests validate Replanner's own logic (attempt
counting, exhaustion, LLM fallback) without depending on pyperplan.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from armdroid.config.schema import ArmPlanningConfig
from armdroid.planning.replanner import Replanner, ReplanningExhaustedError
from armdroid.protocols import PlanStep, SymbolicState

_FIXED_PLAN = [PlanStep(action="move", args=["disk_1", "peg_A", "peg_C"])]


def _make_state() -> SymbolicState:
    return SymbolicState(predicates=frozenset(), objects={})


def _make_mock_planner(plan: list[PlanStep] | None = None) -> MagicMock:
    """Return a mock ArmPlannerProtocol that returns a fixed plan."""
    mock = MagicMock()
    mock.plan.return_value = plan if plan is not None else _FIXED_PLAN
    mock.replan.return_value = plan if plan is not None else _FIXED_PLAN
    return mock


def _make_replanner(
    max_attempts: int = 3,
    llm_enabled: bool = False,
    planner: MagicMock | None = None,
) -> Replanner:
    planning_cfg = ArmPlanningConfig(
        max_replan_attempts=max_attempts,
        llm_replanner_enabled=llm_enabled,
    )
    return Replanner(planning_cfg, planner or _make_mock_planner())


class TestReplanner:
    """Unit tests for Replanner (planner is mocked)."""

    def test_replan_returns_steps(self) -> None:
        replanner = _make_replanner()
        steps = replanner.replan(_make_state(), _make_state(), "grasp failed")
        assert len(steps) > 0
        assert all(isinstance(s, PlanStep) for s in steps)

    def test_replan_exhausted_raises(self) -> None:
        replanner = _make_replanner(max_attempts=1)
        replanner.replan(_make_state(), _make_state(), "error 1")  # attempt 1 OK
        with pytest.raises(ReplanningExhaustedError):
            replanner.replan(_make_state(), _make_state(), "error 2")  # attempt 2 > max

    def test_reset_clears_attempts(self) -> None:
        replanner = _make_replanner(max_attempts=1)
        replanner.replan(_make_state(), _make_state(), "error 1")
        replanner.reset()
        # After reset the counter is at zero — next replan should succeed.
        steps = replanner.replan(_make_state(), _make_state(), "error 2")
        assert len(steps) > 0

    def test_llm_path_falls_back_to_symbolic(self) -> None:
        replanner = _make_replanner(llm_enabled=True)
        steps = replanner.replan(_make_state(), _make_state(), "error")
        assert len(steps) > 0

    def test_attempt_counter_increments(self) -> None:
        replanner = _make_replanner(max_attempts=5)
        for i in range(3):
            replanner.replan(_make_state(), _make_state(), f"error {i}")
        assert replanner._attempt_count == 3
