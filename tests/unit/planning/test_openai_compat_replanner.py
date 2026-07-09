"""Tests for the OpenAI-compatible arm LLM-replanner backend."""

from __future__ import annotations

from typing import Any

import pytest

from armdroid.config.schema import LLMReplannerConfig
from armdroid.domain.state import PlanStep, SymbolicState
from armdroid.planning.llm_replanners.base import (
    LLMReplannerProtocol,
    parse_plan_steps,
)
from armdroid.planning.llm_replanners.openai_compat_backend import (
    OpenAICompatReplanner,
    OpenAICompatSDKMissingError,
)
from tests.helpers.fake_llm_sdk import fake_openai_sdk


def _state(predicates: set[str] | None = None) -> SymbolicState:
    return SymbolicState(
        predicates=frozenset(predicates or {"on(disk_1, peg_A)"}),
        objects={"disk_1": "disk", "peg_A": "peg"},
    )


def test_openai_compat_implements_protocol() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat")
    sdk = fake_openai_sdk("[]")
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    assert isinstance(r, LLMReplannerProtocol)


def test_openai_compat_disabled_returns_empty() -> None:
    cfg = LLMReplannerConfig(enabled=False, backend="openai_compat")
    sdk = fake_openai_sdk('[{"action": "noop", "args": []}]')
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    assert r.replan(_state(), _state(), "fail") == []


def test_openai_compat_disabled_without_sdk_does_not_construct_client() -> None:
    """A configured-but-disabled backend must construct even when the
    ``openai`` extra is absent -- no SDK import, no client, no crash."""
    cfg = LLMReplannerConfig(enabled=False, backend="openai_compat")
    r = OpenAICompatReplanner(cfg)  # no sdk injected; openai not installed
    assert r._client is None  # type: ignore[attr-defined]
    assert r.replan(_state(), _state(), "x") == []


def test_openai_compat_parses_json_response() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat")
    sdk = fake_openai_sdk(
        '[{"action": "move", "args": ["disk_1", "peg_A", "peg_C"]},'
        ' {"action": "open_gripper", "args": []}]'
    )
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    steps = r.replan(_state(), _state({"on(disk_1, peg_C)"}), "grasp failed")
    assert len(steps) == 2
    assert isinstance(steps[0], PlanStep)
    assert steps[0].action == "move"
    assert steps[0].args == ["disk_1", "peg_A", "peg_C"]
    assert steps[1].action == "open_gripper"


def test_openai_compat_non_json_returns_empty() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat")
    sdk = fake_openai_sdk("Sorry, I cannot help with that.")
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    assert r.replan(_state(), _state(), "x") == []


def test_openai_compat_non_list_response_returns_empty() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat")
    sdk = fake_openai_sdk('{"action": "move"}')
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    assert r.replan(_state(), _state(), "x") == []


def test_openai_compat_parses_markdown_fenced_response() -> None:
    """Small open models often wrap JSON in ```json fences despite the
    prompt; the shared parser strips them before decoding."""
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat")
    sdk = fake_openai_sdk('```json\n[{"action": "move", "args": ["a"]}]\n```')
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    steps = r.replan(_state(), _state(), "x")
    assert len(steps) == 1
    assert steps[0].action == "move"


def test_openai_compat_empty_choices_returns_empty() -> None:
    """A response with no choices parses to an empty plan (safe fallback)."""
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat")
    sdk = fake_openai_sdk("[]")
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    assert r._extract_text(object()) == ""  # type: ignore[attr-defined]


def test_openai_compat_null_content_returns_empty() -> None:
    """``message.content is None`` yields an empty string, not a crash."""
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat")
    sdk = fake_openai_sdk("[]")
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    choice = type("C", (), {"message": type("M", (), {"content": None})()})()
    resp = type("R", (), {"choices": [choice]})()
    assert r._extract_text(resp) == ""  # type: ignore[attr-defined]


def _response_with_content(content: object) -> object:
    """Build a minimal chat-completion response with the given content."""
    message = type("M", (), {"content": content})()
    choice = type("C", (), {"message": message})()
    return type("R", (), {"choices": [choice]})()


def test_openai_compat_extract_text_joins_list_dict_parts() -> None:
    """Some gateways return content as a list of ``{"type","text"}`` dicts."""
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat")
    r = OpenAICompatReplanner(cfg, sdk=fake_openai_sdk("[]"))
    resp = _response_with_content(
        [
            {"type": "text", "text": '[{"action": "move",'},
            {"type": "text", "text": ' "args": ["a"]}]'},
            {"type": "image", "text": "SKIP", "url": "http://x"},  # non-text skipped
            {"text": "kept"},  # missing type treated as text
        ]
    )
    assert r._extract_text(resp) == '[{"action": "move", "args": ["a"]}]kept'  # type: ignore[attr-defined]


def test_openai_compat_extract_text_joins_list_object_and_str_parts() -> None:
    """List parts may be objects with ``.text`` or raw strings."""
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat")
    r = OpenAICompatReplanner(cfg, sdk=fake_openai_sdk("[]"))
    part = type("P", (), {"text": "world"})()
    assert r._extract_text(_response_with_content(["hello ", part])) == "hello world"  # type: ignore[attr-defined]


def test_openai_compat_parses_list_shaped_content_end_to_end() -> None:
    """A list-shaped content response parses through replan to steps."""
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat")

    class _ListContentSdk:
        class OpenAI:
            def __init__(self, **kwargs: Any) -> None:
                self.init_kwargs = kwargs
                self.chat = type(
                    "Chat",
                    (),
                    {
                        "completions": type(
                            "Comp",
                            (),
                            {
                                "create": staticmethod(
                                    lambda **kw: _response_with_content(
                                        [{"type": "text", "text": '[{"action": "ok"}]'}]
                                    )
                                )
                            },
                        )()
                    },
                )()

    r = OpenAICompatReplanner(cfg, sdk=_ListContentSdk())
    steps = r.replan(_state(), _state(), "x")
    assert len(steps) == 1
    assert steps[0].action == "ok"


def test_openai_compat_retries_on_exception() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat", max_retries=2)
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


def test_openai_compat_exhausts_retries_and_returns_empty() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat", max_retries=1)
    sdk = fake_openai_sdk("does not matter")
    r = OpenAICompatReplanner(cfg, sdk=sdk)

    def boom(**kwargs: Any) -> Any:
        raise RuntimeError("always fails")

    r._client.chat.completions.create = boom  # type: ignore[assignment]
    assert r.replan(_state(), _state(), "x") == []


def test_openai_compat_passes_config_to_sdk() -> None:
    cfg = LLMReplannerConfig(
        enabled=True,
        backend="openai_compat",
        model="llama-3.1-8b-instant",
        max_tokens=42,
        temperature=0.7,
        system_prompt="You are a robot.",
        request_timeout_s=12.5,
    )
    sdk = fake_openai_sdk("[]")
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    r.replan(_state(), _state(), "fail")
    call = r._client.chat.completions.calls[0]
    assert call["model"] == "llama-3.1-8b-instant"
    assert call["max_tokens"] == 42
    assert call["temperature"] == 0.7
    assert call["timeout"] == 12.5
    # System prompt becomes a leading system message; user turn follows.
    assert call["messages"][0] == {"role": "system", "content": "You are a robot."}
    assert call["messages"][1]["role"] == "user"


def test_openai_compat_omits_system_message_when_prompt_empty() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat", system_prompt="")
    sdk = fake_openai_sdk("[]")
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    r.replan(_state(), _state(), "fail")
    messages = r._client.chat.completions.calls[0]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_openai_compat_wires_base_url_from_api_endpoint() -> None:
    cfg = LLMReplannerConfig(
        enabled=True,
        backend="openai_compat",
        api_endpoint="https://gateway.example/v1",
    )
    sdk = fake_openai_sdk("[]")
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    assert r._client.init_kwargs["base_url"] == "https://gateway.example/v1"


def test_openai_compat_omits_base_url_when_endpoint_empty() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat", api_endpoint="")
    sdk = fake_openai_sdk("[]")
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    assert "base_url" not in r._client.init_kwargs


def test_openai_compat_reads_api_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APEKEY_API_KEY", "sk-test-123")
    cfg = LLMReplannerConfig(
        enabled=True,
        backend="openai_compat",
        api_key_env_var="APEKEY_API_KEY",
    )
    sdk = fake_openai_sdk("[]")
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    assert r._client.init_kwargs["api_key"] == "sk-test-123"


@pytest.mark.asyncio
async def test_openai_compat_areplan_uses_async_client() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat")
    sdk = fake_openai_sdk('[{"action": "ok", "args": []}]', with_async=True)
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    steps = await r.areplan(_state(), _state(), "fail")
    assert len(steps) == 1
    assert steps[0].action == "ok"
    async_client = r._async_client_cache
    assert async_client is not None
    assert len(async_client.chat.completions.calls) == 1


@pytest.mark.asyncio
async def test_openai_compat_areplan_plumbs_config_and_base_url() -> None:
    """The async path reuses the same client kwargs + message builder."""
    cfg = LLMReplannerConfig(
        enabled=True,
        backend="openai_compat",
        model="llama-3.1-8b-instant",
        system_prompt="You are a robot.",
        api_endpoint="https://gateway.example/v1",
    )
    sdk = fake_openai_sdk("[]", with_async=True)
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    await r.areplan(_state(), _state(), "fail")
    async_client = r._async_client_cache
    assert async_client.init_kwargs["base_url"] == "https://gateway.example/v1"
    call = async_client.chat.completions.calls[0]
    assert call["model"] == "llama-3.1-8b-instant"
    assert call["messages"][0] == {"role": "system", "content": "You are a robot."}
    assert call["messages"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_openai_compat_areplan_disabled_returns_empty() -> None:
    cfg = LLMReplannerConfig(enabled=False, backend="openai_compat")
    sdk = fake_openai_sdk("[]", with_async=True)
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    assert await r.areplan(_state(), _state(), "x") == []


@pytest.mark.asyncio
async def test_openai_compat_areplan_reuses_cached_async_client() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat")
    sdk = fake_openai_sdk('[{"action": "ok", "args": []}]', with_async=True)
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    await r.areplan(_state(), _state(), "first")
    first_client = r._async_client_cache
    await r.areplan(_state(), _state(), "second")
    assert r._async_client_cache is first_client


@pytest.mark.asyncio
async def test_openai_compat_areplan_exhausts_retries_returns_empty() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat", max_retries=1)
    sdk = fake_openai_sdk("[]", with_async=True)
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    call_count = {"n": 0}

    async def _always_fail(**kwargs: Any) -> None:
        call_count["n"] += 1
        raise RuntimeError("network down")

    async_client = sdk.AsyncOpenAI()
    async_client.chat.completions.create = _always_fail
    r._async_client_cache = async_client

    result = await r.areplan(_state(), _state(), "fail")
    assert result == []
    assert call_count["n"] == 2  # max_retries=1 -> initial + 1 retry


@pytest.mark.asyncio
async def test_openai_compat_areplan_raises_when_async_missing() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat")
    sdk = fake_openai_sdk("[]", with_async=False)  # no AsyncOpenAI exposed
    r = OpenAICompatReplanner(cfg, sdk=sdk)
    with pytest.raises(OpenAICompatSDKMissingError, match="AsyncOpenAI"):
        await r.areplan(_state(), _state(), "x")


def test_openai_compat_sdk_missing_raises() -> None:
    cfg = LLMReplannerConfig(enabled=True, backend="openai_compat")

    class _BadReplanner(OpenAICompatReplanner):
        def _import_sdk(self) -> Any:  # type: ignore[override]
            raise OpenAICompatSDKMissingError("sdk missing in test")

    with pytest.raises(OpenAICompatSDKMissingError):
        _BadReplanner(cfg)


# ---------------------------------------------------------------------------
# Shared parser: markdown code-fence stripping (relevant to small models)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '[{"action": "move", "args": ["a"]}]',  # bare JSON, no fence
        '```json\n[{"action": "move", "args": ["a"]}]\n```',  # ```json fence
        '```JSON\n[{"action": "move", "args": ["a"]}]\n```',  # upper-case tag
        '```\n[{"action": "move", "args": ["a"]}]\n```',  # fence, no tag
        '```json [{"action": "move", "args": ["a"]}]```',  # single-line fence
        '  ```json\n[{"action": "move", "args": ["a"]}]\n```  ',  # padded
    ],
)
def test_parse_plan_steps_strips_code_fences(text: str) -> None:
    steps = parse_plan_steps(text)
    assert len(steps) == 1
    assert steps[0].action == "move"
    assert steps[0].args == ["a"]


def test_parse_plan_steps_lone_fence_is_empty() -> None:
    """A fence with no JSON body yields no steps rather than raising."""
    assert parse_plan_steps("```json\n```") == []


def test_parse_plan_steps_trailing_prose_after_fence_falls_back_to_empty() -> None:
    """Text with a fence plus trailing prose does not match the whole-string
    fence, so it is parsed as-is and rejected (empty plan)."""
    assert parse_plan_steps('```json\n[{"action": "x"}]\n``` and then more') == []
