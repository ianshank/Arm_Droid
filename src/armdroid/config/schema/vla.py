"""Vision-Language-Action (VLA) agent configuration.

Introduced in Phase A of the Gemini ER 1.6 integration plan (ADR-0007).
Disabled by default so legacy SAC / RSL-RL PPO paths are unaffected.
Bounded fields keep "no hardcoded values" guarantees intact.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ArmVlaConfig(BaseModel):
    """Configuration for a Vision-Language-Action controller backend.

    Disabled by default; when ``enabled`` is True the orchestration
    factory routes ``build_arm_controller`` through the VLA registry
    instead of the SAC / RSL-RL PPO path.
    """

    enabled: bool = Field(
        False,
        description="Enable the VLA control path (vs. the RL agent path).",
    )
    model: str = Field(
        "gemini-robotics-er-1.6",
        min_length=1,
        description="VLA backend model identifier.",
    )
    api_key_env_var: str = Field(
        "GEMINI_API_KEY",
        min_length=1,
        description="Env var holding the VLA backend API key.",
    )
    max_horizon_steps: int = Field(
        16,
        gt=0,
        le=1024,
        description="Maximum action-prediction horizon per ``act`` call.",
    )
    action_clip_rad: float = Field(
        0.25,
        gt=0.0,
        le=3.14159,
        description="Maximum absolute joint delta per step (radians).",
    )
    frame_stack: int = Field(
        4,
        gt=0,
        le=32,
        description="Number of recent RGB frames packed into each request.",
    )
    request_timeout_s: float = Field(
        15.0,
        gt=0.0,
        le=300.0,
        description="Per-request timeout for VLA inference (seconds).",
    )
    max_retries: int = Field(
        2,
        ge=0,
        le=10,
        description="Maximum exponential-backoff retries on transient errors.",
    )


__all__ = ["ArmVlaConfig"]
