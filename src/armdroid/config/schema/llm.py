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
    backend: Literal["null", "llama", "anthropic", "gemini", "openai_compat"] = Field(
        "null",
        description=(
            "Replanner backend selection. ``llama`` and ``openai_compat`` "
            "both select the OpenAI-compatible backend (vLLM, Groq, "
            "Together, OpenRouter, or any gateway speaking that wire "
            "shape); ``openai_compat`` is the protocol-accurate alias."
        ),
    )
    api_endpoint: str = Field(
        "",
        description=(
            "Optional explicit API endpoint URL (the ``base_url`` for the "
            "OpenAI-compatible backend). Empty string ('') means the "
            "backend SDK chooses the default (recommended)."
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
        description=(
            "Max retries on transient errors. Delay between retries is "
            "governed by ``retry_backoff_base_s``."
        ),
    )
    retry_backoff_base_s: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Base delay (seconds) for exponential backoff between retries: "
            "the delay after the Nth failed attempt is "
            "``retry_backoff_base_s * 2**N``. ``0.0`` (default) disables "
            "backoff and retries immediately, preserving legacy behaviour."
        ),
    )


__all__ = ["LLMReplannerConfig"]
