"""LLM replanner sub-configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LLMReplannerConfig(BaseModel):
    """Configuration for the LLM-backed arm replanner.

    Disabled by default; when enabled, ``backend`` selects the concrete
    implementation. ``model``, ``max_tokens``, ``temperature`` and the
    request envelope come from this config so no values are hardcoded
    in the backend modules.
    """

    enabled: bool = Field(
        False,
        description="Enable LLM-backed replanning",
    )
    backend: Literal["null", "llama", "anthropic", "gemini"] = Field(
        "null",
        description="Replanner backend selection",
    )
    api_endpoint: str = Field(
        "",
        description=(
            "Optional explicit API endpoint URL. Empty string ('') means "
            "the backend SDK chooses the default (recommended)."
        ),
    )
    safety_tier: Literal["off", "standard", "strict"] = Field(
        "standard",
        description=(
            "Provider-side safety filter tier. ``off`` disables provider "
            "filters; ``standard`` matches provider default; ``strict`` "
            "applies the most aggressive filtering."
        ),
    )
    transport: Literal["rest", "grpc"] = Field(
        "rest",
        description=(
            "Wire transport for SDK calls. ``rest`` is the safe default; "
            "``grpc`` requires the backend SDK to support it."
        ),
    )
    model: str = Field(
        "claude-sonnet-4-6",
        description="Model identifier passed to the backend",
    )
    max_tokens: int = Field(
        1024,
        gt=0,
        description="Per-request max tokens",
    )
    temperature: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    system_prompt: str = Field(
        "",
        description="System prompt passed to the backend",
    )
    api_key_env_var: str = Field(
        "ANTHROPIC_API_KEY",
        description="Env var holding the API key (Anthropic backend only)",
    )
    request_timeout_s: float = Field(
        30.0,
        gt=0,
        description="Per-request timeout (s)",
    )
    max_retries: int = Field(
        3,
        ge=0,
        description="Max exponential-backoff retries on transient errors",
    )


__all__ = ["LLMReplannerConfig"]
