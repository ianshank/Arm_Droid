"""Unit tests for the Phase A protocol additions.

These tests pin the **disjoint-method invariant** that ADR-0007 relies
on: an instance of ``ArmRLAgentProtocol`` MUST NOT pass ``isinstance``
against ``VisionLanguageAgentProtocol``. Sharing method names between
the two protocols would make the runtime dispatch ambiguous, which is
the exact failure mode peer review caught (B3) in the v1 plan.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import TracebackType

import numpy as np
from numpy.typing import NDArray

from armdroid.domain.protocols import (
    ArmEnvironmentProtocol,
    ArmRLAgentProtocol,
    HighLevelPlannerProtocol,
    InteractionSessionProtocol,
    SafetyGuardProtocol,
    VisionLanguageAgentProtocol,
)
from armdroid.domain.state import (
    ArmAction,
    ArmState,
    InteractionEvent,
    PlanStep,
    SceneInsight,
    Verdict,
)

# ---------------------------------------------------------------------------
# Disjoint-method invariant (peer-review B3)
# ---------------------------------------------------------------------------


class _DummyRLAgent:
    """Minimal stand-in matching ``ArmRLAgentProtocol`` structurally."""

    def build(self, env: ArmEnvironmentProtocol) -> None: ...
    def train(self, total_timesteps: int | None = None) -> None: ...
    def predict(self, observation: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        return np.zeros(6, dtype=np.float64)

    def save(self, path: str | None = None) -> Path:
        return Path("/tmp/x")

    def load(self, path: str) -> None: ...
    @property
    def is_trained(self) -> bool:
        return False

    @property
    def is_built(self) -> bool:
        return False


class _DummyVLA:
    """Minimal stand-in matching ``VisionLanguageAgentProtocol`` structurally."""

    def build_vla(self, env: ArmEnvironmentProtocol) -> None: ...
    def act(
        self,
        rgb: NDArray[np.uint8],
        depth: NDArray[np.float32],
        instruction: str,
        state: ArmState,
    ) -> ArmAction:
        return ArmAction(joint_targets=(0.0,))

    @property
    def is_vla_built(self) -> bool:
        return False

    @property
    def is_vla_loaded(self) -> bool:
        return False


def test_rl_agent_implements_its_protocol() -> None:
    assert isinstance(_DummyRLAgent(), ArmRLAgentProtocol)


def test_vla_agent_implements_its_protocol() -> None:
    assert isinstance(_DummyVLA(), VisionLanguageAgentProtocol)


def test_rl_agent_does_not_match_vla_protocol() -> None:
    """Closes peer-review BLOCKER B3: dispatch must be unambiguous."""
    assert not isinstance(_DummyRLAgent(), VisionLanguageAgentProtocol)


def test_vla_agent_does_not_match_rl_protocol() -> None:
    """Mirror invariant: a VLA-only impl is not silently treated as RL."""
    assert not isinstance(_DummyVLA(), ArmRLAgentProtocol)


# ---------------------------------------------------------------------------
# HighLevelPlannerProtocol
# ---------------------------------------------------------------------------


class _DummyHighLevelPlanner:
    name = "dummy"

    def plan_high_level(self, goal_text: str, scene: SceneInsight | None = None) -> list[PlanStep]:
        return [PlanStep("move", [goal_text])]


def test_high_level_planner_protocol_smoke() -> None:
    planner = _DummyHighLevelPlanner()
    assert isinstance(planner, HighLevelPlannerProtocol)
    steps = planner.plan_high_level("put block in bowl", None)
    assert len(steps) == 1


# ---------------------------------------------------------------------------
# SafetyGuardProtocol
# ---------------------------------------------------------------------------


class _DummySafetyGuard:
    name = "allow_all"

    def check_plan(self, steps: list[PlanStep]) -> Verdict:
        return Verdict(allowed=True, guard_name=self.name)

    def check_action(self, action: ArmAction, state: ArmState) -> Verdict:
        return Verdict(allowed=True, guard_name=self.name)


def test_safety_guard_protocol_smoke() -> None:
    guard = _DummySafetyGuard()
    assert isinstance(guard, SafetyGuardProtocol)
    assert guard.check_plan([]).allowed is True


# ---------------------------------------------------------------------------
# InteractionSessionProtocol
# ---------------------------------------------------------------------------


class _DummyInteractionSession:
    async def __aenter__(self) -> _DummyInteractionSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    async def send_audio(self, chunk: bytes) -> None: ...

    async def send_frame(self, rgb: NDArray[np.uint8]) -> None: ...

    def receive_events(self) -> AsyncIterator[InteractionEvent]:
        async def _gen() -> AsyncIterator[InteractionEvent]:
            yield InteractionEvent(kind="text", text="hi")

        return _gen()


def test_interaction_session_protocol_smoke() -> None:
    session = _DummyInteractionSession()
    assert isinstance(session, InteractionSessionProtocol)


# ---------------------------------------------------------------------------
# Smoke: protocols are exposed in __all__
# ---------------------------------------------------------------------------


def test_all_new_protocols_exported() -> None:
    import armdroid.domain.protocols as protocols_mod

    for name in (
        "VisionLanguageAgentProtocol",
        "SafetyGuardProtocol",
        "InteractionSessionProtocol",
        "HighLevelPlannerProtocol",
    ):
        assert name in protocols_mod.__all__
