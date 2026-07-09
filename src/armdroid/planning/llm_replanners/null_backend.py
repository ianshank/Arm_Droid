"""No-op LLM replanner — returns ``[]`` so the outer ``Replanner`` falls back."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from armdroid.logging.setup import get_logger
from armdroid.planning.llm_replanners.base import LLMReplannerProtocol

if TYPE_CHECKING:
    from armdroid.config.schema import LLMReplannerConfig
    from armdroid.domain.state import PlanStep, SymbolicState

_log = get_logger(__name__)


class NullLLMReplanner:
    """Default replanner: explicitly disabled, always falls back."""

    name = "null"

    @classmethod
    def from_config(
        cls,
        cfg: LLMReplannerConfig,
        *,
        sdk: Any | None = None,
    ) -> NullLLMReplanner:
        """Build a null replanner. ``cfg`` and ``sdk`` are ignored.

        Present so the registry can construct every backend uniformly via
        ``from_config`` regardless of the concrete constructor.
        """
        return cls()

    def replan(
        self,
        current_state: SymbolicState,
        goal_state: SymbolicState,
        error: str,
    ) -> list[PlanStep]:
        _log.debug(
            "null_llm_replan_called",
            error=error,
        )
        return []


_PROTOCOL_CHECK: LLMReplannerProtocol = NullLLMReplanner()
del _PROTOCOL_CHECK


__all__ = ["NullLLMReplanner"]
