"""Task-specific configuration."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class ArmTaskConfig(BaseModel):
    """Task-specific configuration for robot arm manipulation tasks."""

    task_type: Literal["tower_of_hanoi", "laundry_sorting", "pick_place"] = Field(
        "tower_of_hanoi",
        description="Manipulation task type",
    )
    num_disks: int = Field(3, gt=0, le=10, description="Number of disks (Tower of Hanoi)")
    num_pegs: int = Field(3, gt=1, le=5, description="Number of pegs (Tower of Hanoi)")
    peg_positions: list[list[float]] = Field(
        default_factory=lambda: [[0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [0.4, 0.0, 0.0]],
        description="Peg XYZ positions (m) — length must match num_pegs",
    )
    num_baskets: int = Field(3, gt=0, le=5, description="Number of sorting baskets (laundry)")
    basket_positions: list[list[float]] = Field(
        default_factory=lambda: [[0.2, -0.2, 0.0], [0.3, -0.2, 0.0], [0.4, -0.2, 0.0]],
        description="Basket XYZ positions (m) — length must match num_baskets",
    )
    max_episode_steps: int = Field(500, gt=0, description="Max steps per episode")
    num_garments: int = Field(5, gt=0, description="Number of garments per episode (laundry)")

    @model_validator(mode="after")
    def positions_match_count(self) -> Self:
        """Validate position list lengths match counts."""
        if len(self.peg_positions) != self.num_pegs:
            msg = (
                f"peg_positions length ({len(self.peg_positions)})"
                f" must match num_pegs ({self.num_pegs})"
            )
            raise ValueError(msg)
        if len(self.basket_positions) != self.num_baskets:
            msg = (
                f"basket_positions length ({len(self.basket_positions)})"
                f" must match num_baskets ({self.num_baskets})"
            )
            raise ValueError(msg)
        return self


__all__ = ["ArmTaskConfig"]
