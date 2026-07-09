"""Config-driven construction of LLM replanner backends.

Maps :attr:`LLMReplannerConfig.backend` to a concrete
:class:`LLMReplannerProtocol` implementation via a builder mapping (no
``if backend == ...`` ladder -- see the repo convention). Unmapped or
declared-but-unimplemented backends (e.g. ``"gemini"``) resolve to
:class:`NullLLMReplanner` with a warning, so a misconfigured backend
degrades to symbolic planning rather than raising.

A backend whose optional SDK is missing surfaces that as its own
construction error (the concrete backend's lazy import), so an enabled
backend fails fast on a genuine misconfiguration. A configured-but-disabled
backend constructs without the extra installed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from armdroid.logging.setup import get_logger
from armdroid.planning.llm_replanners.anthropic_backend import AnthropicReplanner
from armdroid.planning.llm_replanners.base import LLMReplannerProtocol
from armdroid.planning.llm_replanners.null_backend import NullLLMReplanner
from armdroid.planning.llm_replanners.openai_compat_backend import (
    OpenAICompatReplanner,
)

if TYPE_CHECKING:
    from armdroid.config.schema import LLMReplannerConfig

_log = get_logger(__name__)

_Builder = Callable[["LLMReplannerConfig", Any | None], LLMReplannerProtocol]

# Backend name -> builder. ``"llama"`` and ``"openai_compat"`` both map to
# the OpenAI-compatible backend: ``"llama"`` is the historical config
# literal, ``"openai_compat"`` the protocol-accurate alias.
_BUILDERS: dict[str, _Builder] = {
    "null": lambda cfg, sdk: NullLLMReplanner(),
    "anthropic": lambda cfg, sdk: AnthropicReplanner(cfg, sdk=sdk),
    "llama": lambda cfg, sdk: OpenAICompatReplanner(cfg, sdk=sdk),
    "openai_compat": lambda cfg, sdk: OpenAICompatReplanner(cfg, sdk=sdk),
}


def build_llm_replanner(
    cfg: LLMReplannerConfig,
    *,
    sdk: Any | None = None,
) -> LLMReplannerProtocol:
    """Construct the replanner backend selected by ``cfg.backend``.

    Args:
        cfg: Replanner sub-config; ``cfg.backend`` selects the backend.
        sdk: Optional pre-imported provider SDK forwarded to SDK-backed
            backends (tests inject a fake here). Ignored by the ``null``
            backend.

    Returns:
        A backend implementing :class:`LLMReplannerProtocol`. Unknown or
        unimplemented backend names return :class:`NullLLMReplanner`.
    """
    builder = _BUILDERS.get(cfg.backend)
    if builder is None:
        _log.warning(
            "llm_replanner_backend_unimplemented",
            backend=cfg.backend,
            fallback="null",
        )
        return NullLLMReplanner()
    return builder(cfg, sdk)


__all__ = ["build_llm_replanner"]
