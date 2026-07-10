"""Config-driven construction of LLM replanner backends.

Resolves :attr:`LLMReplannerConfig.backend` against the replanner
:mod:`~armdroid.planning.llm_replanners.registry` (which also discovers
out-of-tree backends via the ``armdroid.llm_replanners`` entry-point
group) and constructs it uniformly through ``from_config``. An unknown or
declared-but-unimplemented backend (e.g. ``"gemini"``) resolves to
:class:`NullLLMReplanner` with a warning, so a misconfigured backend
degrades to symbolic planning rather than raising.

A backend whose optional SDK is missing surfaces that as its own
construction error (the concrete backend's lazy import), so an enabled
backend fails fast on a genuine misconfiguration. A configured-but-disabled
backend constructs without the extra installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from armdroid._registry import RegistryError
from armdroid.logging.setup import get_logger
from armdroid.planning.llm_replanners.null_backend import NullLLMReplanner
from armdroid.planning.llm_replanners.registry import (
    available_llm_replanners,
    get_llm_replanner,
    load_llm_replanner_plugins,
)

if TYPE_CHECKING:
    from armdroid.config.schema import LLMReplannerConfig
    from armdroid.planning.llm_replanners.base import LLMReplannerProtocol

_log = get_logger(__name__)


def build_llm_replanner(
    cfg: LLMReplannerConfig,
    *,
    sdk: Any | None = None,
) -> LLMReplannerProtocol:
    """Construct the replanner backend selected by ``cfg.backend``.

    Args:
        cfg: Replanner sub-config; ``cfg.backend`` selects the backend.
        sdk: Optional pre-imported provider SDK forwarded to the backend
            (tests inject a fake here). Ignored by the ``null`` backend.

    Returns:
        A backend implementing :class:`LLMReplannerProtocol`. Unknown or
        unimplemented backend names return :class:`NullLLMReplanner`.
    """
    # Discover out-of-tree backends once; the registry call is idempotent.
    load_llm_replanner_plugins()
    try:
        backend_cls = get_llm_replanner(cfg.backend)
    except RegistryError as exc:
        _log.warning(
            "llm_replanner_backend_unimplemented",
            backend=cfg.backend,
            fallback="null",
            available=available_llm_replanners(),
            error=str(exc),
        )
        # Fall back through the same uniform construction contract every
        # backend uses, so the null path is not a special case.
        return NullLLMReplanner.from_config(cfg, sdk=sdk)
    _log.debug("llm_replanner_selected", backend=cfg.backend, enabled=cfg.enabled)
    return backend_cls.from_config(cfg, sdk=sdk)


__all__ = ["build_llm_replanner"]
