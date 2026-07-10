"""Tests for the config-driven LLM replanner factory."""

from __future__ import annotations

import pytest

from armdroid.config.schema import LLMReplannerConfig
from armdroid.planning.llm_replanners import build_llm_replanner
from armdroid.planning.llm_replanners.anthropic_backend import AnthropicReplanner
from armdroid.planning.llm_replanners.null_backend import NullLLMReplanner
from armdroid.planning.llm_replanners.openai_compat_backend import (
    OpenAICompatReplanner,
)
from tests.helpers.fake_llm_sdk import fake_anthropic_sdk, fake_openai_sdk


def test_factory_null_backend_returns_null() -> None:
    cfg = LLMReplannerConfig(backend="null")
    assert isinstance(build_llm_replanner(cfg), NullLLMReplanner)


def test_factory_anthropic_backend_returns_anthropic() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="anthropic")
    backend = build_llm_replanner(cfg, sdk=fake_anthropic_sdk("[]"))
    assert isinstance(backend, AnthropicReplanner)


def test_factory_llama_backend_returns_openai_compat() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="llama")
    backend = build_llm_replanner(cfg, sdk=fake_openai_sdk("[]"))
    assert isinstance(backend, OpenAICompatReplanner)


def test_factory_openai_compat_backend_returns_openai_compat() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat")
    backend = build_llm_replanner(cfg, sdk=fake_openai_sdk("[]"))
    assert isinstance(backend, OpenAICompatReplanner)


def test_factory_gemini_backend_falls_back_to_null() -> None:
    """``gemini`` is declared in the config literal but not implemented."""
    cfg = LLMReplannerConfig(enabled=True, backend="gemini")
    assert isinstance(build_llm_replanner(cfg), NullLLMReplanner)


def test_factory_unknown_backend_falls_back_to_null() -> None:
    """A backend value outside the mapping degrades to null, no raise."""
    cfg = LLMReplannerConfig(backend="null")
    # Bypass the schema literal to simulate an unmapped value reaching
    # the factory (e.g. a future literal not yet wired to a builder).
    object.__setattr__(cfg, "backend", "some_future_backend")
    assert isinstance(build_llm_replanner(cfg), NullLLMReplanner)


def test_factory_llama_result_is_functional() -> None:
    """The constructed backend actually parses a response end to end."""
    from armdroid.domain.state import SymbolicState

    cfg = LLMReplannerConfig(enabled=True, backend="llama")
    backend = build_llm_replanner(cfg, sdk=fake_openai_sdk('[{"action": "move", "args": ["a"]}]'))
    state = SymbolicState(predicates=frozenset({"p"}), objects={})
    steps = backend.replan(state, state, "boom")
    assert len(steps) == 1
    assert steps[0].action == "move"


@pytest.mark.parametrize("backend", ["llama", "openai_compat"])
def test_factory_openai_compat_aliases_are_equivalent(backend: str) -> None:
    cfg = LLMReplannerConfig(enabled=True, backend=backend)  # type: ignore[arg-type]
    result = build_llm_replanner(cfg, sdk=fake_openai_sdk("[]"))
    assert isinstance(result, OpenAICompatReplanner)
