"""Swiss-cheese safety middleware configuration (ADR-0009).

Introduced in Phase A of the Gemini ER 1.6 integration plan. The default
``asimov_rule_set="off"`` ensures the chain is permissive until an
operator opts in, satisfying the backwards-compatibility checkpoint that
default rollouts stay behaviourally unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ArmSafetyConfig(BaseModel):
    """Bounded kinematic and policy limits consumed by the safety chain.

    Every field is a Pydantic-bounded scalar so no guard implementation
    can leak a hardcoded threshold. Paths default to repo-relative
    locations and are configurable via YAML overlay or ``ARMDROID_*``
    env vars.
    """

    asimov_rule_set: Literal["off", "lab", "strict", "custom"] = Field(
        "off",
        description=(
            "Asimov-style adjudication rule set. ``off`` disables the "
            "natural-language guard entirely; ``lab`` and ``strict`` use "
            "the bundled NEISS corpus at decreasing thresholds; "
            "``custom`` reads ``neiss_corpus_path``."
        ),
    )
    max_tcp_speed_mps: float = Field(
        0.5,
        gt=0.0,
        le=5.0,
        description="ISO TS 15066-derived TCP speed cap (m/s).",
    )
    max_joint_velocity_rad_s: float = Field(
        1.5,
        gt=0.0,
        le=10.0,
        description="Per-joint angular velocity cap (rad/s).",
    )
    fragile_force_n: float = Field(
        5.0,
        gt=0.0,
        le=200.0,
        description="Maximum contact force allowed against fragile objects (N).",
    )
    neiss_corpus_path: Path = Field(
        Path("assets/safety/neiss_corpus.json"),
        description=(
            "Path to the NEISS-derived injury corpus consumed by the "
            "Asimov guard. Packaged inside the wheel; override for "
            "operator-supplied corpora."
        ),
    )
    chain_deny_overrides: bool = Field(
        True,
        description=(
            "Compose chained guards with deny-overrides semantics "
            "(any single ``allowed=False`` short-circuits the chain). "
            "Always True today; field reserved for future allow-all "
            "diagnostic modes."
        ),
    )


__all__ = ["ArmSafetyConfig"]
