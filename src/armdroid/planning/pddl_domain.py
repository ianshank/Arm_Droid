"""PDDL domain and problem generator for Tower of Hanoi.

Generates valid PDDL domain and problem files from the current
symbolic state and task configuration. Guarantees optimal solutions
with exactly 2^n - 1 moves for n disks.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from armdroid.domain.state import SymbolicState
from armdroid.logging.setup import get_logger

if TYPE_CHECKING:
    from armdroid.config.schema import ArmTaskConfig

_log = get_logger(__name__)

# Matches Python/functional notation: "on(disk_1, peg_a)"
_FUNC_PRED_RE = re.compile(r"^(\w+)\(([^)]*)\)$")


def _predicate_to_pddl_atom(pred: str) -> str:
    """Normalise a predicate string to a PDDL init-section atom.

    Handles three input forms:

    * PDDL form  ``(on disk_1 peg_a)``  → returned as-is.
    * Functional ``on(disk_1, peg_a)``   → ``(on disk_1 peg_a)``.
    * Space-sep  ``on disk_1 peg_a``     → ``(on disk_1 peg_a)``.

    Args:
        pred: A predicate string in any recognised form.

    Returns:
        A PDDL atom string starting and ending with parentheses.
    """
    pred = pred.strip()
    if pred.startswith("("):
        return pred  # Already PDDL form.
    m = _FUNC_PRED_RE.match(pred)
    if m:
        name = m.group(1)
        args = [a.strip() for a in m.group(2).split(",") if a.strip()]
        return f"({name} {' '.join(args)})"
    # Space-separated "name arg1 arg2 …"
    return f"({pred})"


HANOI_DOMAIN_PDDL = """\
(define (domain tower-of-hanoi)
  (:requirements :strips)
  (:predicates
    (on ?x ?y)
    (clear ?x)
    (smaller ?x ?y)
    (disk ?x)
    (peg ?x)
  )
  (:action move
    :parameters (?disk ?from ?to)
    :precondition (and
      (disk ?disk)
      (clear ?disk)
      (clear ?to)
      (on ?disk ?from)
      (smaller ?disk ?to)
    )
    :effect (and
      (on ?disk ?to)
      (clear ?from)
      (not (on ?disk ?from))
      (not (clear ?to))
    )
  )
)
"""


def generate_domain() -> str:
    """Generate the Tower of Hanoi PDDL domain definition.

    Returns:
        PDDL domain string.
    """
    return HANOI_DOMAIN_PDDL


def generate_problem(
    task_cfg: ArmTaskConfig,
    initial_state: SymbolicState | None = None,
) -> str:
    """Generate a Tower of Hanoi PDDL problem from task config.

    If no initial state is provided, generates the standard initial
    state with all disks stacked on peg_A in order.

    Args:
        task_cfg: Task config with num_disks and num_pegs.
        initial_state: Optional current symbolic state to plan from.

    Returns:
        PDDL problem string.
    """
    num_disks = task_cfg.num_disks
    num_pegs = task_cfg.num_pegs

    disk_names = [f"disk_{i + 1}" for i in range(num_disks)]
    # Lower-case peg names (peg_a, peg_b, …) match the token casing that
    # pyperplan normalises all identifiers to, keeping solution names consistent
    # with the input domain regardless of the planner backend used.
    peg_names = [f"peg_{chr(97 + i)}" for i in range(num_pegs)]

    # Objects
    objects = " ".join(disk_names) + " " + " ".join(peg_names)

    # Invariant facts: type predicates (disk/peg) and size ordering (smaller).
    # These are static throughout any Tower of Hanoi problem — they never
    # change regardless of which disk is where — so they belong in every
    # :init section, whether we're planning from a concrete perceived state or
    # from the canonical starting position.
    invariant_lines: list[str] = []
    for name in disk_names:
        invariant_lines.append(f"(disk {name})")
    for name in peg_names:
        invariant_lines.append(f"(peg {name})")
    # Smaller-than: disk_1 (index 0) is smallest; each disk is smaller than
    # all later disks and all pegs.
    all_objects = disk_names + peg_names
    for i, d in enumerate(disk_names):
        for j in range(i + 1, len(all_objects)):
            invariant_lines.append(f"(smaller {d} {all_objects[j]})")

    # Initial state predicates.
    # Treat None *and* an empty predicate set the same: both mean "start from
    # the canonical Tower of Hanoi position".  An empty predicate set is never
    # a valid planning state (no on/clear fluents) and would cause pyperplan
    # to return no solution even with the invariants present.
    init_lines: list[str] = list(invariant_lines)
    if initial_state is not None and initial_state.predicates:
        # Caller supplied concrete fluents — convert to PDDL atom syntax.
        # StateExtractor emits Python-style "on(disk_1, peg_a)" notation;
        # PDDL needs "(on disk_1 peg_a)".  Handle both forms.
        for pred in sorted(initial_state.predicates):
            init_lines.append(_predicate_to_pddl_atom(pred))
    else:
        # Default: all disks stacked on peg_a, smallest on top.
        init_lines.append(f"(on {disk_names[-1]} {peg_names[0]})")
        for i in range(num_disks - 2, -1, -1):
            init_lines.append(f"(on {disk_names[i]} {disk_names[i + 1]})")

        # Clear: top disk and all pegs except peg_a
        init_lines.append(f"(clear {disk_names[0]})")
        for name in peg_names[1:]:
            init_lines.append(f"(clear {name})")

    init_preds = "\n    ".join(init_lines)

    # Goal: all disks on last peg
    goal_lines: list[str] = []
    goal_lines.append(f"(on {disk_names[-1]} {peg_names[-1]})")
    for i in range(num_disks - 2, -1, -1):
        goal_lines.append(f"(on {disk_names[i]} {disk_names[i + 1]})")
    goal_preds = "\n      ".join(goal_lines)

    problem = f"""\
(define (problem hanoi-{num_disks}disk)
  (:domain tower-of-hanoi)
  (:objects {objects})
  (:init
    {init_preds}
  )
  (:goal (and
      {goal_preds}
    )
  )
)
"""
    _log.debug("pddl_problem_generated", num_disks=num_disks, num_pegs=num_pegs)
    return problem


def optimal_move_count(num_disks: int) -> int:
    """Calculate optimal move count for Tower of Hanoi.

    Args:
        num_disks: Number of disks.

    Returns:
        Optimal number of moves: 2^n - 1.
    """
    return (1 << num_disks) - 1
