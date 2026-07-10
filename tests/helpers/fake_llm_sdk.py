"""Reusable fake LLM SDK harness for unit tests.

Mirrors the shape that the ``anthropic`` and ``google-genai`` SDKs both
follow at the boundary armdroid actually consumes: a ``Client`` exposing
``messages.create`` (and optionally an ``AsyncClient`` whose ``create``
is awaitable). One canned ``response_text`` is returned per call so the
test can assert structural properties of the parse path without involving
any real network round-trip.

The helper is intentionally provider-agnostic: it parametrises the
client-class name (``client_class``) and an optional async sibling
(``async_client_class``) so a single helper covers Anthropic, Gemini, and
any future LLM backend that follows the same shape. Callers that want
the historical ``{"Anthropic": ..., "AsyncAnthropic": ...}`` surface use
the :func:`fake_anthropic_sdk` convenience wrapper.

The returned ``SimpleNamespace`` is what the production backends receive
through their ``sdk`` constructor kwarg, so this helper completely
bypasses the lazy ``_import_sdk`` path.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

__all__ = [
    "fake_anthropic_sdk",
    "fake_gemini_sdk",
    "fake_llm_sdk",
    "fake_openai_sdk",
]


class _Block:
    """Single content block, matching the ``content[*].type == "text"`` shape."""

    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Response:
    """Container holding one or more content blocks."""

    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


class _SyncMessages:
    """Sync ``messages.create`` recording call kwargs for assertions."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        return _Response(self._response_text)


class _AsyncMessages:
    """Async ``messages.create`` recording call kwargs for assertions."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        return _Response(self._response_text)


def _make_client_class(messages_factory: Any) -> type:
    """Build a ``Client`` class whose ``messages`` attribute is fresh per instance."""

    class _Client:
        def __init__(self, api_key: str | None = None) -> None:
            self.api_key = api_key
            self.messages = messages_factory()

    return _Client


def fake_llm_sdk(
    response_text: str,
    *,
    client_class: str,
    async_client_class: str | None = None,
) -> SimpleNamespace:
    """Build a minimal stand-in for any LLM SDK that follows the Anthropic shape.

    Args:
        response_text: Canned text the fake returns from ``messages.create``.
        client_class: Attribute name to expose the sync client under
            (``"Anthropic"`` for the anthropic SDK, ``"Client"`` for
            ``google-genai``).
        async_client_class: Optional attribute name for the async sibling
            (``"AsyncAnthropic"``); when ``None``, the returned namespace
            has no async client surface.

    Returns:
        A :class:`types.SimpleNamespace` with the requested class
        attributes. Pass this to a backend constructor's ``sdk=`` kwarg.
    """
    members: dict[str, Any] = {
        client_class: _make_client_class(lambda: _SyncMessages(response_text)),
    }
    if async_client_class is not None:
        members[async_client_class] = _make_client_class(lambda: _AsyncMessages(response_text))
    return SimpleNamespace(**members)


def fake_anthropic_sdk(response_text: str, *, with_async: bool = False) -> SimpleNamespace:
    """Convenience wrapper for the anthropic SDK shape used by ``AnthropicReplanner``."""
    return fake_llm_sdk(
        response_text,
        client_class="Anthropic",
        async_client_class="AsyncAnthropic" if with_async else None,
    )


def fake_gemini_sdk(response_text: str, *, with_async: bool = False) -> SimpleNamespace:
    """Convenience wrapper for the google-genai SDK shape used by ``GeminiReplanner``."""
    return fake_llm_sdk(
        response_text,
        client_class="Client",
        async_client_class="AsyncClient" if with_async else None,
    )


# ---------------------------------------------------------------------------
# OpenAI chat-completions shape (``choices[0].message.content``)
# ---------------------------------------------------------------------------


class _OAMessage:
    """Assistant message, matching ``choices[*].message.content``."""

    def __init__(self, text: str) -> None:
        self.role = "assistant"
        self.content = text


class _OAChoice:
    """Single choice wrapping one assistant message."""

    def __init__(self, text: str) -> None:
        self.message = _OAMessage(text)


class _OAResponse:
    """Chat completion response holding one choice."""

    def __init__(self, text: str) -> None:
        self.choices = [_OAChoice(text)]


class _OASyncCompletions:
    """Sync ``chat.completions.create`` recording call kwargs."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _OAResponse:
        self.calls.append(kwargs)
        return _OAResponse(self._response_text)


class _OAAsyncCompletions:
    """Async ``chat.completions.create`` recording call kwargs."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _OAResponse:
        self.calls.append(kwargs)
        return _OAResponse(self._response_text)


def _make_openai_client_class(completions_factory: Any) -> type:
    """Build an OpenAI-shaped client exposing ``chat.completions.create``.

    Records the constructor kwargs (``api_key``, ``base_url``) on the
    instance so tests can assert the backend plumbed config through.
    """

    class _OAClient:
        def __init__(self, **kwargs: Any) -> None:
            self.init_kwargs = kwargs
            self.chat = SimpleNamespace(completions=completions_factory())

    return _OAClient


def fake_openai_sdk(response_text: str, *, with_async: bool = False) -> SimpleNamespace:
    """Stand-in for the ``openai`` SDK used by ``OpenAICompatReplanner``.

    Exposes ``OpenAI`` (and optionally ``AsyncOpenAI``) client classes
    whose ``chat.completions.create`` returns a canned response and
    records call kwargs. Pass to the backend's ``sdk=`` constructor kwarg.
    """
    members: dict[str, Any] = {
        "OpenAI": _make_openai_client_class(lambda: _OASyncCompletions(response_text)),
    }
    if with_async:
        members["AsyncOpenAI"] = _make_openai_client_class(
            lambda: _OAAsyncCompletions(response_text)
        )
    return SimpleNamespace(**members)
