"""Bidirectional interaction session configuration (Gemini Live API).

Introduced in Phase A of the Gemini ER 1.6 integration plan; consumed by
Phase E's ``InteractionSessionProtocol`` implementations. Disabled by
default so the orchestrator's rollout loop continues to run with a
``NullInteractionSession``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ArmInteractionConfig(BaseModel):
    """Configuration for a bidirectional audio/video interaction session.

    Canonical "enable" flag lives here (not on ``LLMReplannerConfig``)
    to avoid duplicating truth-sources for "is Live API on?" - peer
    review M2.
    """

    enabled: bool = Field(
        False,
        description="Enable a bidirectional interaction session at orchestrator startup.",
    )
    backend: str = Field(
        "null",
        min_length=1,
        description=(
            "Registry name of the interaction backend "
            "(``null`` or ``gemini_live``). Resolved via the "
            "``armdroid.interaction_sessions`` entry-point group."
        ),
    )
    api_key_env_var: str = Field(
        "GEMINI_API_KEY",
        min_length=1,
        description="Env var holding the interaction backend API key.",
    )
    audio_sample_rate_hz: int = Field(
        16000,
        gt=0,
        le=96000,
        description="PCM audio sample rate forwarded upstream (Hz).",
    )
    video_fps: float = Field(
        5.0,
        gt=0.0,
        le=60.0,
        description="Video frames per second forwarded upstream.",
    )
    session_timeout_s: float = Field(
        300.0,
        gt=0.0,
        le=3600.0,
        description="Maximum session duration before forced close (seconds).",
    )


__all__ = ["ArmInteractionConfig"]
