"""Unit tests for ``ArmAction`` and ``SceneInsight`` value objects."""

from __future__ import annotations

import numpy as np
import pytest

from armdroid.domain.state import ArmAction, SceneInsight

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
