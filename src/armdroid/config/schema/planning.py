"""Symbolic planning configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from armdroid.config.schema.llm import LLMReplannerConfig


class ArmPlanningConfig(BaseModel):
    """Symbolic planning configuration for robot arm tasks."""

    pddl_domain_path: Path = Field(
        Path("planning/pddl/hanoi_domain.pddl"),
        description="PDDL domain file path",
    )
    planner_backend: Literal["pyperplan", "fast_downward"] = Field(
        "pyperplan",
        description="PDDL solver backend",
    )
    pyperplan_search: Literal["bfs", "astar", "wastar", "gbf"] = Field(
        "bfs",
        description=(
            "Pyperplan search algorithm. 'bfs' guarantees shortest plan; "
            "'astar'/'wastar'/'gbf' use heuristic search (hadd) and are "
            "faster on larger problems."
        ),
    )
    llm_replanner_enabled: bool = Field(
        False,
        description="Enable LLM-based adaptive replanning on execution failure",
    )
    max_replan_attempts: int = Field(3, gt=0, description="Max replanning attempts before abort")
    planning_timeout_s: float = Field(5.0, gt=0, description="Maximum planning time (s)")
    llm_replanner: LLMReplannerConfig | None = Field(
        None,
        description="LLM-backed replanner config (None=use legacy symbolic fallback)",
    )


__all__ = ["ArmPlanningConfig"]
