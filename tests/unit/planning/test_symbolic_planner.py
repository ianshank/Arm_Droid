"""Tests for symbolic planner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from armdroid.config.schema import ArmPlanningConfig, ArmTaskConfig
from armdroid.planning.pddl_domain import optimal_move_count
from armdroid.planning.symbolic_planner import PlanningError, SymbolicPlanner
from armdroid.protocols import SymbolicState


def _make_planner(num_disks: int = 3) -> SymbolicPlanner:
    """Create a planner with default configs."""
    planning_cfg = ArmPlanningConfig()
    task_cfg = ArmTaskConfig(num_disks=num_disks, num_pegs=3)
    return SymbolicPlanner(planning_cfg, task_cfg)


def _make_initial_state() -> SymbolicState:
    """Create dummy initial state."""
    return SymbolicState(predicates=frozenset(), objects={})


def _make_goal_state() -> SymbolicState:
    """Create dummy goal state."""
    return SymbolicState(predicates=frozenset(), objects={})


class TestRecursiveSolver:
    """Test the recursive fallback solver (guaranteed optimal)."""

    def test_1_disk_produces_1_move(self) -> None:
        planner = _make_planner(num_disks=1)
        steps = planner._solve_recursive()
        assert len(steps) == 1
        assert steps[0].action == "move"

    def test_3_disks_produces_7_moves(self) -> None:
        planner = _make_planner(num_disks=3)
        steps = planner._solve_recursive()
        assert len(steps) == optimal_move_count(3)

    def test_5_disks_produces_31_moves(self) -> None:
        planner = _make_planner(num_disks=5)
        steps = planner._solve_recursive()
        assert len(steps) == optimal_move_count(5)

    def test_moves_reference_correct_pegs(self) -> None:
        planner = _make_planner(num_disks=3)
        steps = planner._solve_recursive()
        peg_names = {"peg_a", "peg_b", "peg_c"}
        for step in steps:
            assert step.args[1] in peg_names  # source peg
            assert step.args[2] in peg_names  # target peg

    def test_no_disk_moved_from_same_to_same(self) -> None:
        planner = _make_planner(num_disks=3)
        steps = planner._solve_recursive()
        for step in steps:
            assert step.args[1] != step.args[2]  # source != target


class TestSymbolicPlannerPlan:
    """Test the plan() method."""

    def test_plan_returns_steps(self) -> None:
        planner = _make_planner(num_disks=3)
        steps = planner.plan(_make_initial_state(), _make_goal_state())
        assert len(steps) > 0

    def test_replan_returns_steps(self) -> None:
        planner = _make_planner(num_disks=3)
        steps = planner.replan(_make_initial_state(), _make_goal_state(), "test error")
        assert len(steps) > 0


def _make_mock_planner_module(
    return_value: list[MagicMock] | None = None,
    side_effect: Exception | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Build a mock for the ``pyperplan.planner`` sub-module.

    The production code does ``from pyperplan import planner as _pio_planner``
    (a lazy import inside ``_solve_pddl``).  Patching via ``sys.modules``
    intercepts that import without touching module-level state.
    """
    mock_planner = MagicMock()
    mock_planner.SEARCHES = {"bfs": MagicMock(), "astar": MagicMock()}
    mock_planner.HEURISTICS = {"hadd": MagicMock()}
    if side_effect is not None:
        mock_planner.search_plan.side_effect = side_effect
    else:
        mock_planner.search_plan.return_value = return_value

    mock_pkg = MagicMock()
    mock_pkg.planner = mock_planner
    return mock_pkg, mock_planner


def _make_operator(name: str) -> MagicMock:
    """Return a mock pyperplan ``Operator`` with the given name string."""
    op = MagicMock()
    op.name = name
    return op


class TestPyperplanIntegration:
    """Test _solve_pddl and _parse_solution with mocked pyperplan.

    The production code does ``from pyperplan import planner as _pio_planner``
    inside ``_solve_pddl`` (lazy import).  We intercept it via
    ``patch.dict("sys.modules", ...)`` so the mock takes effect regardless of
    whether pyperplan is installed in the test environment.
    """

    def test_solve_pddl_with_mocked_pyperplan(self) -> None:
        planner = _make_planner(num_disks=2)

        ops = [
            _make_operator("(move disk_1 peg_a peg_c)"),
            _make_operator("(move disk_2 peg_a peg_b)"),
        ]
        mock_pkg, mock_planner = _make_mock_planner_module(return_value=ops)
        with patch.dict("sys.modules", {"pyperplan": mock_pkg, "pyperplan.planner": mock_planner}):
            steps = planner._solve_pddl("(domain)", "(problem)")

        assert len(steps) == 2
        assert steps[0].action == "move"
        assert steps[0].args == ["disk_1", "peg_a", "peg_c"]
        assert steps[1].args == ["disk_2", "peg_a", "peg_b"]

    def test_solve_pddl_pyperplan_returns_none(self) -> None:
        planner = _make_planner(num_disks=2)

        mock_pkg, mock_planner = _make_mock_planner_module(return_value=None)
        with (
            patch.dict("sys.modules", {"pyperplan": mock_pkg, "pyperplan.planner": mock_planner}),
            pytest.raises(PlanningError, match="no solution"),
        ):
            planner._solve_pddl("(domain)", "(problem)")

    def test_plan_uses_pyperplan_when_available(self) -> None:
        planner = _make_planner(num_disks=2)

        ops = [_make_operator("(move disk_1 peg_a peg_b)")]
        mock_pkg, mock_planner = _make_mock_planner_module(return_value=ops)
        with patch.dict("sys.modules", {"pyperplan": mock_pkg, "pyperplan.planner": mock_planner}):
            steps = planner.plan(_make_initial_state(), _make_goal_state())

        assert len(steps) == 1
        mock_planner.search_plan.assert_called_once()

    def test_plan_wraps_pyperplan_exception(self) -> None:
        planner = _make_planner(num_disks=2)

        mock_pkg, mock_planner = _make_mock_planner_module(side_effect=RuntimeError("solver crash"))
        with (
            patch.dict("sys.modules", {"pyperplan": mock_pkg, "pyperplan.planner": mock_planner}),
            pytest.raises(PlanningError, match="Planning failed"),
        ):
            planner.plan(_make_initial_state(), _make_goal_state())


class TestParseSolution:
    """Test _parse_solution directly."""

    def test_parses_pddl_actions(self) -> None:
        planner = _make_planner(num_disks=2)
        solution = [
            "(move disk_1 peg_A peg_C)",
            "(move disk_2 peg_A peg_B)",
            "(move disk_1 peg_C peg_B)",
        ]
        steps = planner._parse_solution(solution)
        assert len(steps) == 3
        assert steps[0].action == "move"
        assert steps[2].args == ["disk_1", "peg_C", "peg_B"]

    def test_handles_empty_lines(self) -> None:
        planner = _make_planner(num_disks=2)
        solution = ["(move disk_1 peg_A peg_C)", "", "  "]
        steps = planner._parse_solution(solution)
        assert len(steps) == 1

    def test_handles_single_action(self) -> None:
        planner = _make_planner(num_disks=1)
        solution = ["(move disk_1 peg_A peg_C)"]
        steps = planner._parse_solution(solution)
        assert len(steps) == 1
        assert steps[0].action == "move"
