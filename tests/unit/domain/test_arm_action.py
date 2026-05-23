"""Unit tests for ``ArmAction`` and ``SceneInsight`` value objects."""

from __future__ import annotations

import numpy as np
import pytest

from armdroid.domain.state import ArmAction, InteractionEvent, SceneInsight, Verdict

# ---------------------------------------------------------------------------
# ArmAction
# ---------------------------------------------------------------------------


def test_arm_action_constructs_with_defaults() -> None:
    action = ArmAction(joint_targets=(0.1, 0.2, 0.3))
    assert action.joint_targets == (0.1, 0.2, 0.3)
    assert action.gripper == 0.0
    assert action.timestamp_s is None


def test_arm_action_is_frozen() -> None:
    action = ArmAction(joint_targets=(0.0,))
    with pytest.raises((AttributeError, TypeError)):
        action.gripper = 0.5  # type: ignore[misc]


def test_arm_action_is_slotted() -> None:
    """frozen+slots: ``__slots__`` is declared and no ``__dict__`` exists."""
    assert ArmAction.__slots__ == ("joint_targets", "gripper", "timestamp_s")
    action = ArmAction(joint_targets=(0.0,))
    assert not hasattr(action, "__dict__")


def test_arm_action_equality_by_value() -> None:
    a = ArmAction(joint_targets=(1.0, 2.0), gripper=0.5, timestamp_s=10.0)
    b = ArmAction(joint_targets=(1.0, 2.0), gripper=0.5, timestamp_s=10.0)
    c = ArmAction(joint_targets=(1.0, 2.0), gripper=0.6, timestamp_s=10.0)
    assert a == b
    assert a != c


def test_arm_action_hashable() -> None:
    a = ArmAction(joint_targets=(1.0, 2.0))
    assert hash(a) == hash(ArmAction(joint_targets=(1.0, 2.0)))


def test_arm_action_from_array_converts_ndarray() -> None:
    arr = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    action = ArmAction.from_array(arr, gripper=0.7, timestamp_s=42.0)
    assert action.joint_targets == (0.1, 0.2, 0.3)
    assert action.gripper == 0.7
    assert action.timestamp_s == 42.0


def test_arm_action_from_array_handles_2d_input() -> None:
    arr = np.array([[0.1, 0.2]], dtype=np.float64)
    action = ArmAction.from_array(arr)
    assert action.joint_targets == (0.1, 0.2)


def test_arm_action_from_array_handles_empty_array() -> None:
    """Empty NDArray must materialise as an empty joints tuple, not raise."""
    arr = np.array([], dtype=np.float64)
    action = ArmAction.from_array(arr)
    assert action.joint_targets == ()
    assert action.gripper == 0.0


def test_arm_action_from_array_preserves_nan_and_inf() -> None:
    """Down-stream guards (Phase F) reject NaN/Inf; the value object must
    not silently drop them so the guard can do its job."""
    arr = np.array([np.nan, np.inf, -np.inf], dtype=np.float64)
    action = ArmAction.from_array(arr)
    # tuple comparison with NaN: each NaN compares unequal to itself, so
    # check structurally via isnan / isinf rather than tuple equality.
    assert len(action.joint_targets) == 3
    assert np.isnan(action.joint_targets[0])
    assert np.isposinf(action.joint_targets[1])
    assert np.isneginf(action.joint_targets[2])


def test_arm_action_from_array_accepts_python_floats() -> None:
    """``np.asarray`` accepts list[float] - belt-and-braces for callers
    that pre-convert without realising ``from_array`` already handles it."""
    action = ArmAction.from_array(np.asarray([1.0, 2.0]))
    assert action.joint_targets == (1.0, 2.0)


# ---------------------------------------------------------------------------
# SceneInsight
# ---------------------------------------------------------------------------


def test_scene_insight_defaults() -> None:
    si = SceneInsight()
    assert si.crops == ()
    assert si.rotations_deg == ()
    assert si.notes == ""


def test_scene_insight_frozen_and_equal() -> None:
    si_a = SceneInsight(crops=((0, 0, 10, 10),), rotations_deg=(90.0,), notes="x")
    si_b = SceneInsight(crops=((0, 0, 10, 10),), rotations_deg=(90.0,), notes="x")
    assert si_a == si_b
    with pytest.raises((AttributeError, TypeError)):
        si_a.notes = "mutated"  # type: ignore[misc]


def test_scene_insight_hashable() -> None:
    si = SceneInsight(crops=((0, 0, 1, 1),), rotations_deg=(0.0,), notes="n")
    assert hash(si) == hash(SceneInsight(crops=((0, 0, 1, 1),), rotations_deg=(0.0,), notes="n"))


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def test_verdict_defaults_permissive() -> None:
    """``Verdict(allowed=True)`` requires no further fields - keeps guard
    construction ergonomic in the default-allow case (ADR-0009)."""
    v = Verdict(allowed=True)
    assert v.allowed is True
    assert v.reason == ""
    assert v.guard_name == ""


def test_verdict_deny_carries_reason_and_name() -> None:
    v = Verdict(allowed=False, reason="velocity cap exceeded", guard_name="iso_kinematic")
    assert v.allowed is False
    assert v.reason == "velocity cap exceeded"
    assert v.guard_name == "iso_kinematic"


def test_verdict_frozen() -> None:
    v = Verdict(allowed=True)
    with pytest.raises((AttributeError, TypeError)):
        v.allowed = False  # type: ignore[misc]


def test_verdict_equality_and_hash() -> None:
    a = Verdict(allowed=False, reason="x", guard_name="g")
    b = Verdict(allowed=False, reason="x", guard_name="g")
    assert a == b
    assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# InteractionEvent
# ---------------------------------------------------------------------------


def test_interaction_event_defaults() -> None:
    e = InteractionEvent(kind="audio")
    assert e.kind == "audio"
    assert e.text == ""
    assert e.timestamp_s is None


def test_interaction_event_all_kinds_construct() -> None:
    """Pin the documented kind vocabulary so Phase E backends don't
    accidentally introduce undocumented event types."""
    for kind in ("audio", "text", "frame", "replan_request", "session_end"):
        e = InteractionEvent(kind=kind, text="payload", timestamp_s=1.0)
        assert e.kind == kind


def test_interaction_event_frozen() -> None:
    e = InteractionEvent(kind="text")
    with pytest.raises((AttributeError, TypeError)):
        e.text = "mutated"  # type: ignore[misc]
