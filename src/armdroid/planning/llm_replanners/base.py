"""Protocol and shared helpers for LLM-backed arm replanners.

The :class:`LLMReplannerProtocol` defines the surface every backend
implements. The two module-level helpers below are provider-neutral: the
replan prompt and the JSON-array-to-:class:`PlanStep` parse are identical
regardless of which LLM answered, so every backend (Anthropic,
OpenAI-compatible, future ones) shares them instead of duplicating the
logic. Backends keep thin private wrappers that delegate here so their
existing call sites and tests stay stable.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from armdroid.domain.state import PlanStep
from armdroid.logging.setup import get_logger

if TYPE_CHECKING:
    from armdroid.config.schema import LLMReplannerConfig
    from armdroid.domain.state import SymbolicState

_log = get_logger(__name__)

# Matches a whole markdown code fence, capturing the body: an opening
# ``` optionally followed by a language tag (``json``), the content, and
# a closing ```. Small open models frequently wrap JSON in these despite
# the prompt asking for a bare array, so the parser strips them first.
_CODE_FENCE_RE = re.compile(r"^```[A-Za-z0-9_]*\n?(?P<body>.*?)\n?```$", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Return ``text`` with a surrounding markdown code fence removed.

    Leaves text without a fence untouched. Only a fence that wraps the
    entire (stripped) string is removed; trailing prose after a closing
    fence means no match, and the raw text is returned for the parser to
    reject in the usual way.
    """
    stripped = text.strip()
    match = _CODE_FENCE_RE.match(stripped)
    if match is None:
        return stripped
    _log.debug("llm_replan_stripped_code_fence")
    return match.group("body").strip()


def backoff_delay_s(attempt: int, base_s: float) -> float:
    """Return the exponential-backoff delay (seconds) for a retry attempt.

    ``attempt`` is the 0-indexed attempt that just failed, so the delay
    doubles each round: ``base_s * 2**attempt``. A non-positive
    ``base_s`` disables backoff and returns ``0.0`` -- the historical
    "retry immediately" behaviour and the default, which also keeps unit
    tests free of real sleeps. The value is config-driven
    (:attr:`LLMReplannerConfig.retry_backoff_base_s`); no delay is
    hardcoded in the backends.
    """
    if base_s <= 0.0 or attempt < 0:
        return 0.0
    return base_s * (2.0**attempt)


@runtime_checkable
class LLMReplannerProtocol(Protocol):
    """Generates a recovery plan from a (state, goal, error) triple.

    Implementations may call any LLM provider (Anthropic, an
    OpenAI-compatible gateway, an in-process stub for tests, ...).
    Returning an empty list is the explicit signal for the outer
    :class:`Replanner` to fall back to its symbolic planner -- no other
    "no plan" sentinel is used.
    """

    @property
    def name(self) -> str: ...

    def replan(
        self,
        current_state: SymbolicState,
        goal_state: SymbolicState,
        error: str,
    ) -> list[PlanStep]: ...


class LLMReplannerBackend(LLMReplannerProtocol, Protocol):
    """A replanner that can be constructed uniformly from config.

    Extends :class:`LLMReplannerProtocol` with a ``from_config``
    classmethod so the registry can build any backend the same way --
    ``cls.from_config(cfg, sdk=sdk)`` -- regardless of the concrete
    constructor. This is the contract out-of-tree backends implement to
    be discoverable via the ``armdroid.llm_replanners`` entry-point group.
    """

    @classmethod
    def from_config(
        cls,
        cfg: LLMReplannerConfig,
        *,
        sdk: Any | None = None,
    ) -> LLMReplannerProtocol: ...


def build_replan_prompt(
    current_state: SymbolicState,
    goal_state: SymbolicState,
    error: str,
) -> str:
    """Build the provider-neutral user prompt for a replan request.

    The instruction to reply with ONLY a JSON array is deliberate: the
    parse in :func:`parse_plan_steps` expects a bare array and returns
    an empty list on any deviation, which the outer replanner treats as
    "fall back to symbolic planning".
    """
    return (
        "You are replanning a failed robot arm action.\n\n"
        f"Current symbolic state predicates: {sorted(current_state.predicates)}\n"
        f"Goal symbolic state predicates: {sorted(goal_state.predicates)}\n"
        f"Failure description: {error}\n\n"
        "Reply with a JSON array of plan steps. Each step is an object "
        "with keys 'action' (str) and optional 'args' (list[str]). Reply "
        "with ONLY the JSON array -- no surrounding prose."
    )


def parse_plan_steps(text: str) -> list[PlanStep]:
    """Parse an LLM response into a list of :class:`PlanStep`.

    Tolerant by design: empty text, a markdown-fenced payload, non-JSON
    text, a non-list top level, or malformed entries all resolve to a
    (possibly partial) list without raising. An empty result signals the
    outer replanner to fall back to symbolic planning, so a bad response
    degrades safely.
    """
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return []
    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError:
        _log.warning("llm_replan_non_json_response", preview=cleaned[:120])
        return []
    if not isinstance(raw, list):
        _log.warning("llm_replan_non_list_response")
        return []
    steps: list[PlanStep] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        action = entry.get("action")
        if not isinstance(action, str):
            continue
        args = entry.get("args", []) or []
        if not isinstance(args, list):
            continue
        steps.append(PlanStep(action=action, args=[str(a) for a in args]))
    return steps


__all__ = [
    "LLMReplannerBackend",
    "LLMReplannerProtocol",
    "backoff_delay_s",
    "build_replan_prompt",
    "parse_plan_steps",
]
