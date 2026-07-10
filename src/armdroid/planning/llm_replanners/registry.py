"""LLM replanner registry -- plugin seam for replanner backends.

Mirrors the other subsystem registries (``armdroid.hardware.registry``,
``armdroid.planning.registry``, ...): built-in backends register on
import, and out-of-tree backends plug in via the
``armdroid.llm_replanners`` entry-point group. The registry stores backend
*classes* (identity-stable, so entry-point re-registration is idempotent);
each class implements :meth:`LLMReplannerBackend.from_config` so callers
construct any backend uniformly via ``cls.from_config(cfg, sdk=sdk)``.

``llama`` and ``openai_compat`` are both registered to the
OpenAI-compatible backend -- ``llama`` is the historical config literal,
``openai_compat`` the protocol-accurate alias.
"""

from __future__ import annotations

from armdroid._registry import Registry
from armdroid.logging.setup import get_logger
from armdroid.planning.llm_replanners.anthropic_backend import AnthropicReplanner
from armdroid.planning.llm_replanners.base import LLMReplannerBackend
from armdroid.planning.llm_replanners.null_backend import NullLLMReplanner
from armdroid.planning.llm_replanners.openai_compat_backend import (
    OpenAICompatReplanner,
)

_log = get_logger(__name__)

_REPLANNERS: Registry[type[LLMReplannerBackend]] = Registry(
    kind="llm_replanner",
    entry_point_group="armdroid.llm_replanners",
)

_REPLANNERS.register("null", NullLLMReplanner)
_REPLANNERS.register("anthropic", AnthropicReplanner)
_REPLANNERS.register("llama", OpenAICompatReplanner)
_REPLANNERS.register("openai_compat", OpenAICompatReplanner)


def register_llm_replanner(name: str, backend: type[LLMReplannerBackend]) -> None:
    """Register a replanner backend class under ``name``."""
    _REPLANNERS.register(name, backend)


def get_llm_replanner(name: str) -> type[LLMReplannerBackend]:
    """Return the replanner backend class registered under ``name``.

    Raises:
        RegistryError: If ``name`` is not registered.
    """
    return _REPLANNERS.get(name)


def available_llm_replanners() -> list[str]:
    """Return the sorted list of registered replanner backend names."""
    return _REPLANNERS.available()


def load_llm_replanner_plugins() -> int:
    """Discover and register out-of-tree replanner backends via entry points."""
    return _REPLANNERS.load_entry_points()


__all__ = [
    "available_llm_replanners",
    "get_llm_replanner",
    "load_llm_replanner_plugins",
    "register_llm_replanner",
]
