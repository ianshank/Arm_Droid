"""OpenAI-compatible arm replanner (vLLM / Groq / Together / gateways).

Speaks the OpenAI ``chat/completions`` wire shape, so it drives any
endpoint that implements that surface -- a local vLLM/llama.cpp server, a
hosted gateway, or providers such as Groq, Together, or OpenRouter. No
provider name is hardcoded: the target is chosen entirely by config via
:attr:`LLMReplannerConfig.api_endpoint` (the ``base_url``) and
:attr:`LLMReplannerConfig.api_key_env_var`, so the same backend covers
every OpenAI-compatible host.

The ``openai`` Python SDK is an OPTIONAL dependency -- install with
``pip install -e ".[openai]"``. The import lives inside
:meth:`OpenAICompatReplanner._import_sdk` so the rest of the harness
imports without the dependency present.

All knobs (model, max_tokens, temperature, system_prompt, request
timeout, max retries, api key env var, base URL) come from
:class:`armdroid.config.schema.LLMReplannerConfig`. The backend never
hardcodes a model name, prompt, endpoint, or timeout.

Two surfaces are exposed, mirroring :class:`AnthropicReplanner`:

* :meth:`OpenAICompatReplanner.replan` is **synchronous** and conforms
  to :class:`LLMReplannerProtocol`. Callers inside an event loop MUST
  wrap it in :func:`asyncio.to_thread` to avoid blocking the loop.
* :meth:`OpenAICompatReplanner.areplan` is **async-native** -- it lazily
  instantiates :class:`openai.AsyncOpenAI` and awaits the request.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from armdroid.logging.setup import get_logger
from armdroid.planning.llm_replanners.base import (
    arun_replan_with_retries,
    build_replan_prompt,
    run_replan_with_retries,
)

if TYPE_CHECKING:
    from armdroid.config.schema import LLMReplannerConfig
    from armdroid.domain.state import PlanStep, SymbolicState

_log = get_logger(__name__)


class OpenAICompatSDKMissingError(RuntimeError):
    """Raised when the ``openai`` SDK is not installed."""


class OpenAICompatReplanner:
    """LLM replanner that calls an OpenAI-compatible chat completions API.

    The replan method is synchronous to match the existing ``Replanner``
    surface; it submits a single non-streaming request and parses the
    JSON response into :class:`PlanStep` instances. Failures are logged
    and surfaced as an empty list so the outer ``Replanner`` can fall
    back to its symbolic planner.
    """

    name = "openai_compat"

    @classmethod
    def from_config(
        cls,
        cfg: LLMReplannerConfig,
        *,
        sdk: Any | None = None,
    ) -> OpenAICompatReplanner:
        """Construct from config (registry/plugin entry point)."""
        return cls(cfg, sdk=sdk)

    def __init__(
        self,
        cfg: LLMReplannerConfig,
        *,
        sdk: Any | None = None,
    ) -> None:
        """Build the OpenAI-compatible replanner.

        Args:
            cfg: Replanner config -- supplies model, prompt, timeouts,
                ``api_endpoint`` (base URL) and the api-key env var name.
            sdk: Optional pre-imported ``openai`` module (for tests). When
                ``None``, the constructor lazily imports ``openai`` and
                raises :class:`OpenAICompatSDKMissingError` if unavailable.
        """
        self._cfg = cfg
        self._sdk: Any = sdk
        self._client: Any = None
        self._async_client_cache: Any | None = None
        # Only import the optional ``openai`` SDK and build the client when
        # the backend is actually enabled. A configured-but-disabled backend
        # (e.g. ``backend: openai_compat`` with ``enabled: false``) must not
        # fail to construct just because the extra is not installed; fail-fast
        # is reserved for the enabled path where the SDK is genuinely needed.
        if cfg.enabled:
            if self._sdk is None:
                self._sdk = self._import_sdk()
            self._client = self._build_client(cfg, self._sdk)
            _log.info(
                "openai_compat_replanner_initialised",
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                timeout_s=cfg.request_timeout_s,
                base_url=cfg.api_endpoint or "<sdk-default>",
            )
        else:
            _log.debug("openai_compat_replanner_constructed_disabled")

    # -------------------------------------------------------- public API
    def replan(
        self,
        current_state: SymbolicState,
        goal_state: SymbolicState,
        error: str,
    ) -> list[PlanStep]:
        """Synchronous replan -- blocks the calling thread on HTTP I/O.

        Conforms to :class:`LLMReplannerProtocol`. Async callers should
        use :meth:`areplan` (or wrap this in :func:`asyncio.to_thread`)
        so the event loop is not blocked during the network round-trip.
        """
        if not self._cfg.enabled:
            _log.debug("openai_compat_replanner_disabled")
            return []
        messages = self._build_messages(current_state, goal_state, error)

        def _request() -> str:
            response = self._client.chat.completions.create(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                messages=messages,
                timeout=self._cfg.request_timeout_s,
            )
            return self._extract_text(response)

        return run_replan_with_retries(
            _request,
            cfg=self._cfg,
            log=_log,
            event_prefix="openai_compat_replanner",
        )

    async def areplan(
        self,
        current_state: SymbolicState,
        goal_state: SymbolicState,
        error: str,
    ) -> list[PlanStep]:
        """Async-native replan -- never blocks the event loop.

        Lazily instantiates :class:`openai.AsyncOpenAI` on first use and
        awaits the network call. Reuses the same retry, parsing, and
        logging logic as :meth:`replan`.
        """
        if not self._cfg.enabled:
            _log.debug("openai_compat_replanner_disabled")
            return []
        client = self._async_client()
        messages = self._build_messages(current_state, goal_state, error)

        async def _request() -> str:
            response = await client.chat.completions.create(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                messages=messages,
                timeout=self._cfg.request_timeout_s,
            )
            return self._extract_text(response)

        return await arun_replan_with_retries(
            _request,
            cfg=self._cfg,
            log=_log,
            event_prefix="openai_compat_replanner_async",
        )

    def _async_client(self) -> Any:
        """Lazily build an ``AsyncOpenAI`` client on first use."""
        if self._async_client_cache is not None:
            return self._async_client_cache
        async_cls = getattr(self._sdk, "AsyncOpenAI", None)
        if async_cls is None:
            msg = (
                "The installed 'openai' SDK does not expose AsyncOpenAI; "
                "upgrade to a recent release to use "
                "OpenAICompatReplanner.areplan."
            )
            raise OpenAICompatSDKMissingError(msg)
        self._async_client_cache = async_cls(**self._client_kwargs(self._cfg))
        return self._async_client_cache

    # ------------------------------------------------------------ helpers
    def _import_sdk(self) -> Any:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - environment-dependent
            msg = (
                "The 'openai' SDK is not installed. Install with "
                '`pip install -e ".[openai]"` to enable this backend.'
            )
            raise OpenAICompatSDKMissingError(msg) from exc
        return openai

    def _client_kwargs(self, cfg: LLMReplannerConfig) -> dict[str, Any]:
        """Resolve constructor kwargs shared by the sync and async clients.

        ``base_url`` is only passed when ``api_endpoint`` is a non-empty
        string so the SDK default (the OpenAI public endpoint) still
        applies when no gateway is configured. ``api_key`` is read from
        the configured env var; an empty value is passed through as
        ``None`` so the SDK can apply its own resolution.
        """
        api_key = os.environ.get(cfg.api_key_env_var, "")
        kwargs: dict[str, Any] = {"api_key": api_key or None}
        if cfg.api_endpoint:
            kwargs["base_url"] = cfg.api_endpoint
        return kwargs

    def _build_client(self, cfg: LLMReplannerConfig, sdk: Any) -> Any:
        return sdk.OpenAI(**self._client_kwargs(cfg))

    def _build_messages(
        self,
        current_state: SymbolicState,
        goal_state: SymbolicState,
        error: str,
    ) -> list[dict[str, str]]:
        """Build the chat ``messages`` list.

        The OpenAI shape has no dedicated ``system=`` kwarg, so a
        non-empty ``system_prompt`` becomes a leading ``system`` message.
        The user turn reuses the provider-neutral replan prompt.
        """
        messages: list[dict[str, str]] = []
        if self._cfg.system_prompt:
            messages.append({"role": "system", "content": self._cfg.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": build_replan_prompt(current_state, goal_state, error),
            }
        )
        return messages

    def _extract_text(self, response: Any) -> str:
        """Pull the assistant text from an OpenAI chat completion response.

        Reads ``choices[0].message.content``. ``content`` is usually a
        plain string, but some OpenAI-compatible gateways return it as a
        list of content parts (``{"type": "text", "text": ...}`` dicts or
        objects with a ``.text`` attribute); both shapes are handled. A
        missing/empty ``choices`` or a content shape that yields no text
        returns an empty string (which :func:`parse_plan_steps` turns into
        an empty plan) and emits a debug log so an operator can tell "the
        model answered in an unexpected shape" apart from "the model
        genuinely returned an empty plan".
        """
        choices = getattr(response, "choices", None) or []
        if not choices:
            _log.debug("openai_compat_replanner_no_choices")
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text = self._join_content_parts(content)
            if text:
                return text
        _log.debug(
            "openai_compat_replanner_nonstr_content",
            content_type=type(content).__name__,
        )
        return ""

    @staticmethod
    def _join_content_parts(parts: list[Any]) -> str:
        """Concatenate the text of a list-shaped ``message.content``.

        Each part may be a raw string, a ``{"type": "text", "text": ...}``
        dict, or an object exposing a ``.text`` attribute. Parts whose
        ``type`` is present and is not ``"text"`` (e.g. images) are
        skipped even if they carry a ``text`` field; a missing ``type`` is
        treated as text for backward compatibility.
        """
        chunks: list[str] = []
        for part in parts:
            if isinstance(part, str):
                chunks.append(part)
                continue
            if isinstance(part, dict):
                part_type = part.get("type", "text")
                text = part.get("text")
            else:
                part_type = getattr(part, "type", "text")
                text = getattr(part, "text", None)
            if part_type == "text" and isinstance(text, str):
                chunks.append(text)
        return "".join(chunks).strip()


__all__ = ["OpenAICompatReplanner", "OpenAICompatSDKMissingError"]
