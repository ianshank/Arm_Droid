"""Anthropic-backed arm replanner.

The ``anthropic`` Python SDK is an OPTIONAL dependency — install with
``pip install -e ".[anthropic]"``. The import lives inside
:meth:`AnthropicReplanner._client` so the rest of the harness can be
imported without the dependency present.

All knobs (model, max_tokens, temperature, system_prompt, request
timeout, max retries, api key env var) come from
:class:`armdroid.config.schema.LLMReplannerConfig`. The backend never
hardcodes a model name, prompt, or timeout.

Two surfaces are exposed:

* :meth:`AnthropicReplanner.replan` is **synchronous** and conforms to
  :class:`LLMReplannerProtocol`. It performs a blocking HTTP request via
  the official sync Anthropic SDK. Callers running inside an event loop
  MUST wrap it in :func:`asyncio.to_thread` to avoid blocking the loop.
* :meth:`AnthropicReplanner.areplan` is **async-native** — it lazily
  instantiates :class:`anthropic.AsyncAnthropic` and awaits the request,
  so async callers can await it directly without thread-hopping.

The arm planning pipeline currently consumes the sync surface; the
async surface is provided so hot-path callers (orchestrator hooks, MCP
bridges, sub-agents) can opt into the non-blocking variant.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from armdroid.logging.setup import get_logger
from armdroid.planning.llm_replanners.base import (
    arun_replan_with_retries,
    build_replan_prompt,
    parse_plan_steps,
    run_replan_with_retries,
)

if TYPE_CHECKING:
    from armdroid.config.schema import LLMReplannerConfig
    from armdroid.domain.state import PlanStep, SymbolicState

_log = get_logger(__name__)


class AnthropicSDKMissingError(RuntimeError):
    """Raised when the ``anthropic`` SDK is not installed."""


class AnthropicReplanner:
    """LLM replanner that calls the Anthropic Messages API.

    The replan method is synchronous to match the existing ``Replanner``
    surface; it submits a single non-streaming request and parses the
    JSON response into :class:`PlanStep` instances. Failures are logged
    and surfaced as an empty list so the outer ``Replanner`` can fall
    back to its symbolic planner.
    """

    name = "anthropic"

    @classmethod
    def from_config(
        cls,
        cfg: LLMReplannerConfig,
        *,
        sdk: Any | None = None,
    ) -> AnthropicReplanner:
        """Construct from config (registry/plugin entry point)."""
        return cls(cfg, sdk=sdk)

    def __init__(
        self,
        cfg: LLMReplannerConfig,
        *,
        sdk: Any | None = None,
    ) -> None:
        """Build the Anthropic replanner.

        Args:
            cfg: Replanner config — supplies model, prompt, timeouts, etc.
            sdk: Optional pre-imported ``anthropic`` module (for tests).
                When ``None``, the constructor lazily imports ``anthropic``
                and raises :class:`AnthropicSDKMissingError` if unavailable.
        """
        self._cfg = cfg
        self._sdk = sdk if sdk is not None else self._import_sdk()
        self._client = self._build_client(cfg, self._sdk)
        self._async_client_cache: Any | None = None
        _log.info(
            "anthropic_replanner_initialised",
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            timeout_s=cfg.request_timeout_s,
        )

    # -------------------------------------------------------- public API
    def replan(
        self,
        current_state: SymbolicState,
        goal_state: SymbolicState,
        error: str,
    ) -> list[PlanStep]:
        """Synchronous replan — blocks the calling thread on HTTP I/O.

        Conforms to :class:`LLMReplannerProtocol`. Async callers should
        use :meth:`areplan` (or wrap this in :func:`asyncio.to_thread`)
        so the event loop is not blocked during the network round-trip.
        """
        if not self._cfg.enabled:
            _log.debug("anthropic_replanner_disabled")
            return []
        prompt = self._build_user_prompt(current_state, goal_state, error)

        def _request() -> str:
            response = self._client.messages.create(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                system=self._cfg.system_prompt,
                messages=[{"role": "user", "content": prompt}],
                timeout=self._cfg.request_timeout_s,
            )
            return self._extract_text(response)

        return run_replan_with_retries(
            _request,
            cfg=self._cfg,
            log=_log,
            event_prefix="anthropic_replanner",
        )

    async def areplan(
        self,
        current_state: SymbolicState,
        goal_state: SymbolicState,
        error: str,
    ) -> list[PlanStep]:
        """Async-native replan — never blocks the event loop.

        Lazily instantiates :class:`anthropic.AsyncAnthropic` on first use
        and awaits the network call. Reuses the same retry, parsing, and
        logging logic as :meth:`replan`. Available for async callers that
        want to surface the LLM replanner from a hot-path coroutine
        without wrapping in :func:`asyncio.to_thread`.
        """
        if not self._cfg.enabled:
            _log.debug("anthropic_replanner_disabled")
            return []
        client = self._async_client()
        prompt = self._build_user_prompt(current_state, goal_state, error)

        async def _request() -> str:
            response = await client.messages.create(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                system=self._cfg.system_prompt,
                messages=[{"role": "user", "content": prompt}],
                timeout=self._cfg.request_timeout_s,
            )
            return self._extract_text(response)

        return await arun_replan_with_retries(
            _request,
            cfg=self._cfg,
            log=_log,
            event_prefix="anthropic_replanner_async",
        )

    def _async_client(self) -> Any:
        """Lazily build an ``AsyncAnthropic`` client on first use."""
        if self._async_client_cache is not None:
            return self._async_client_cache
        api_key = os.environ.get(self._cfg.api_key_env_var, "")
        async_cls = getattr(self._sdk, "AsyncAnthropic", None)
        if async_cls is None:
            msg = (
                "The installed 'anthropic' SDK does not expose AsyncAnthropic; "
                "upgrade to a recent release to use AnthropicReplanner.areplan."
            )
            raise AnthropicSDKMissingError(msg)
        self._async_client_cache = async_cls(api_key=api_key or None)
        return self._async_client_cache

    # ------------------------------------------------------------ helpers
    def _import_sdk(self) -> Any:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment-dependent
            msg = (
                "The 'anthropic' SDK is not installed. Install with "
                '`pip install -e ".[anthropic]"` to enable this backend.'
            )
            raise AnthropicSDKMissingError(msg) from exc
        return anthropic

    def _build_client(self, cfg: LLMReplannerConfig, sdk: Any) -> Any:
        api_key = os.environ.get(cfg.api_key_env_var, "")
        # Most Anthropic SDK versions accept api_key=None and pull from env;
        # we pass the resolved value explicitly so config drives behaviour.
        return sdk.Anthropic(api_key=api_key or None)

    def _build_user_prompt(
        self,
        current_state: SymbolicState,
        goal_state: SymbolicState,
        error: str,
    ) -> str:
        return build_replan_prompt(current_state, goal_state, error)

    def _extract_text(self, response: Any) -> str:
        # The Messages API returns ``content`` as a list of blocks, each
        # with a ``.text`` attribute for ``type='text'`` blocks.
        content = getattr(response, "content", None) or []
        chunks: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks).strip()

    def _parse_steps(self, text: str) -> list[PlanStep]:
        return parse_plan_steps(text)


__all__ = ["AnthropicReplanner", "AnthropicSDKMissingError"]
