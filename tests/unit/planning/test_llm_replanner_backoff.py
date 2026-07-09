"""Tests for config-driven exponential backoff between replanner retries."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from armdroid.config.schema import LLMReplannerConfig
from armdroid.domain.state import SymbolicState
from armdroid.planning.llm_replanners.anthropic_backend import AnthropicReplanner
from armdroid.planning.llm_replanners.base import backoff_delay_s
from armdroid.planning.llm_replanners.openai_compat_backend import (
    OpenAICompatReplanner,
)
from tests.helpers.fake_llm_sdk import fake_anthropic_sdk, fake_openai_sdk


def _state() -> SymbolicState:
    return SymbolicState(predicates=frozenset({"p"}), objects={})


# ---------------------------------------------------------------------------
# Pure helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("attempt", "base", "expected"),
    [
        (0, 0.0, 0.0),  # disabled
        (5, 0.0, 0.0),  # disabled regardless of attempt
        (0, 0.5, 0.5),  # base * 2**0
        (1, 0.5, 1.0),  # base * 2**1
        (2, 0.5, 2.0),  # base * 2**2
        (3, 1.0, 8.0),  # base * 2**3
        (-1, 0.5, 0.0),  # guard against negative attempt
    ],
)
def test_backoff_delay_s(attempt: int, base: float, expected: float) -> None:
    assert backoff_delay_s(attempt, base) == expected


# ---------------------------------------------------------------------------
# Sync backend honours backoff via time.sleep
# ---------------------------------------------------------------------------


def test_sync_backoff_sleeps_between_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    cfg = LLMReplannerConfig(
        enabled=True,
        backend="openai_compat",
        max_retries=2,
        retry_backoff_base_s=0.5,
    )
    sdk = fake_openai_sdk('[{"action": "ok", "args": []}]')
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    real_create = r._client.chat.completions.create
    failures = {"left": 1}

    def flaky(**kwargs: Any) -> Any:
        if failures["left"] > 0:
            failures["left"] -= 1
            raise RuntimeError("network")
        return real_create(**kwargs)

    r._client.chat.completions.create = flaky  # type: ignore[assignment]
    steps = r.replan(_state(), _state(), "x")
    assert len(steps) == 1
    assert slept == [0.5]  # one failure -> one backoff of base * 2**0


def test_sync_backoff_skips_sleep_on_final_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    cfg = LLMReplannerConfig(
        enabled=True,
        backend="openai_compat",
        max_retries=2,
        retry_backoff_base_s=0.5,
    )
    sdk = fake_openai_sdk("[]")
    r = OpenAICompatReplanner(cfg, sdk=sdk)

    def boom(**kwargs: Any) -> Any:
        raise RuntimeError("always fails")

    r._client.chat.completions.create = boom  # type: ignore[assignment]
    assert r.replan(_state(), _state(), "x") == []
    # 3 attempts (0,1,2); sleeps only after attempts 0 and 1, never the last.
    assert slept == [0.5, 1.0]


def test_sync_no_backoff_when_base_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    cfg = LLMReplannerConfig(
        enabled=True, backend="openai_compat", max_retries=2
    )  # retry_backoff_base_s defaults to 0.0
    sdk = fake_openai_sdk("[]")
    r = OpenAICompatReplanner(cfg, sdk=sdk)

    def boom(**kwargs: Any) -> Any:
        raise RuntimeError("always fails")

    r._client.chat.completions.create = boom  # type: ignore[assignment]
    assert r.replan(_state(), _state(), "x") == []
    assert slept == []  # default disables backoff -> immediate retries


# ---------------------------------------------------------------------------
# Async backend honours backoff via asyncio.sleep
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_backoff_awaits_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []

    async def fake_sleep(s: float) -> None:
        slept.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    cfg = LLMReplannerConfig(
        enabled=True,
        backend="openai_compat",
        max_retries=1,
        retry_backoff_base_s=0.25,
    )
    sdk = fake_openai_sdk("[]", with_async=True)
    r = OpenAICompatReplanner(cfg, sdk=sdk)

    async def _always_fail(**kwargs: Any) -> None:
        raise RuntimeError("network down")

    async_client = sdk.AsyncOpenAI()
    async_client.chat.completions.create = _always_fail
    r._async_client_cache = async_client

    assert await r.areplan(_state(), _state(), "x") == []
    # max_retries=1 -> attempts 0,1; sleep only after attempt 0.
    assert slept == [0.25]


# ---------------------------------------------------------------------------
# The Anthropic backend shares the same backoff mechanism
# ---------------------------------------------------------------------------


def test_anthropic_sync_backoff_sleeps_between_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    cfg = LLMReplannerConfig(
        enabled=True,
        backend="anthropic",
        max_retries=2,
        retry_backoff_base_s=0.5,
    )
    sdk = fake_anthropic_sdk('[{"action": "ok", "args": []}]')
    r = AnthropicReplanner(cfg, sdk=sdk)
    real_create = r._client.messages.create
    failures = {"left": 1}

    def flaky(**kwargs: Any) -> Any:
        if failures["left"] > 0:
            failures["left"] -= 1
            raise RuntimeError("network")
        return real_create(**kwargs)

    r._client.messages.create = flaky  # type: ignore[assignment]
    steps = r.replan(_state(), _state(), "x")
    assert len(steps) == 1
    assert slept == [0.5]


@pytest.mark.asyncio
async def test_anthropic_async_backoff_awaits_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []

    async def fake_sleep(s: float) -> None:
        slept.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    cfg = LLMReplannerConfig(
        enabled=True,
        backend="anthropic",
        max_retries=1,
        retry_backoff_base_s=0.25,
    )
    sdk = fake_anthropic_sdk("[]", with_async=True)
    r = AnthropicReplanner(cfg, sdk=sdk)

    async def _always_fail(**kwargs: Any) -> None:
        raise RuntimeError("network down")

    async_client = sdk.AsyncAnthropic()
    async_client.messages.create = _always_fail
    r._async_client_cache = async_client

    assert await r.areplan(_state(), _state(), "x") == []
    assert slept == [0.25]
