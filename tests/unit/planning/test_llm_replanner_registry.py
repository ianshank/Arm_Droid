"""Tests for the LLM replanner registry + uniform ``from_config`` contract."""

from __future__ import annotations

import pytest

from armdroid._registry import RegistryError
from armdroid.config.schema import LLMReplannerConfig
from armdroid.planning.llm_replanners import (
    available_llm_replanners,
    get_llm_replanner,
    load_llm_replanner_plugins,
    register_llm_replanner,
)
from armdroid.planning.llm_replanners.anthropic_backend import AnthropicReplanner
from armdroid.planning.llm_replanners.base import LLMReplannerProtocol
from armdroid.planning.llm_replanners.null_backend import NullLLMReplanner
from armdroid.planning.llm_replanners.openai_compat_backend import (
    OpenAICompatReplanner,
)
from tests.helpers.fake_llm_sdk import fake_anthropic_sdk, fake_openai_sdk


def test_builtin_backends_are_registered() -> None:
    names = available_llm_replanners()
    assert {"null", "anthropic", "llama", "openai_compat"} <= set(names)


def test_get_returns_expected_classes() -> None:
    assert get_llm_replanner("null") is NullLLMReplanner
    assert get_llm_replanner("anthropic") is AnthropicReplanner
    assert get_llm_replanner("llama") is OpenAICompatReplanner
    assert get_llm_replanner("openai_compat") is OpenAICompatReplanner


def test_get_unknown_raises_registry_error() -> None:
    with pytest.raises(RegistryError):
        get_llm_replanner("no_such_backend")


def test_load_plugins_is_idempotent_and_returns_int() -> None:
    # No test entry points installed under the group -> 0 newly loaded,
    # and calling twice must not raise (idempotent).
    first = load_llm_replanner_plugins()
    second = load_llm_replanner_plugins()
    assert isinstance(first, int)
    assert second == 0


def test_register_custom_backend_roundtrips() -> None:
    class _CustomReplanner:
        name = "custom"

        @classmethod
        def from_config(cls, cfg: object, *, sdk: object = None) -> _CustomReplanner:
            return cls()

        def replan(self, *_args: object, **_kwargs: object) -> list:
            return []

    register_llm_replanner("custom_test_backend", _CustomReplanner)  # type: ignore[arg-type]
    try:
        assert get_llm_replanner("custom_test_backend") is _CustomReplanner
        assert "custom_test_backend" in available_llm_replanners()
    finally:
        # Keep global registry state clean for other tests.
        from armdroid.planning.llm_replanners.registry import _REPLANNERS

        _REPLANNERS.unregister("custom_test_backend")


# ---------------------------------------------------------------------------
# Uniform from_config construction contract
# ---------------------------------------------------------------------------


def test_null_from_config_ignores_args() -> None:
    cfg = LLMReplannerConfig(backend="null")
    backend = NullLLMReplanner.from_config(cfg, sdk=object())
    assert isinstance(backend, NullLLMReplanner)
    assert isinstance(backend, LLMReplannerProtocol)


def test_anthropic_from_config_builds_backend() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="anthropic")
    backend = AnthropicReplanner.from_config(cfg, sdk=fake_anthropic_sdk("[]"))
    assert isinstance(backend, AnthropicReplanner)


def test_openai_compat_from_config_builds_backend() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat")
    backend = OpenAICompatReplanner.from_config(cfg, sdk=fake_openai_sdk("[]"))
    assert isinstance(backend, OpenAICompatReplanner)
