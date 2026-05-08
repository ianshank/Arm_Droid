"""Tests for protocol data classes (SymbolicState, PlanStep, DetectedObject)."""

from __future__ import annotations

import numpy as np
import pytest

from armdroid.protocols import DetectedObject, PlanStep, SymbolicState


class TestSymbolicState:
    """SymbolicState equality, hashing, and slot behaviour."""

    def test_equal_states(self) -> None:
        a = SymbolicState(predicates=frozenset({"on(d1,peg_A)"}), objects={"d1": "disk"})
        b = SymbolicState(predicates=frozenset({"on(d1,peg_A)"}), objects={"d1": "disk"})
        assert a == b

    def test_unequal_states_different_predicates(self) -> None:
        a = SymbolicState(predicates=frozenset({"on(d1,peg_A)"}), objects={"d1": "disk"})
        b = SymbolicState(predicates=frozenset({"on(d1,peg_B)"}), objects={"d1": "disk"})
        assert a != b

    def test_unequal_states_different_objects(self) -> None:
        a = SymbolicState(predicates=frozenset(), objects={"d1": "disk"})
        b = SymbolicState(predicates=frozenset(), objects={"d2": "disk"})
        assert a != b

    def test_eq_returns_not_implemented_for_other_type(self) -> None:
        s = SymbolicState(predicates=frozenset(), objects={})
        assert s.__eq__("not a state") is NotImplemented

    def test_hash_equal_for_same_predicates(self) -> None:
        a = SymbolicState(predicates=frozenset({"p"}), objects={})
        b = SymbolicState(predicates=frozenset({"p"}), objects={})
        assert hash(a) == hash(b)

    def test_usable_as_dict_key(self) -> None:
        s = SymbolicState(predicates=frozenset({"p"}), objects={})
        d = {s: "value"}
        assert d[s] == "value"


class TestPlanStep:
    """PlanStep construction and repr."""

    def test_repr_with_args(self) -> None:
        step = PlanStep("move", ["disk1", "peg_A", "peg_C"])
        assert repr(step) == "move(disk1, peg_A, peg_C)"

    def test_repr_no_args(self) -> None:
        step = PlanStep("home", [])
        assert repr(step) == "home()"

    def test_attributes_set(self) -> None:
        step = PlanStep("grasp", ["disk1"])
        assert step.action == "grasp"
        assert step.args == ["disk1"]


class TestDetectedObject:
    """DetectedObject construction."""

    def test_attributes_set(self) -> None:
        pos = np.array([0.1, 0.2, 0.3])
        ori = np.zeros(3)
        bbox = np.array([10.0, 20.0, 50.0, 80.0])
        obj = DetectedObject("id1", "disk_1", 0.95, pos, ori, bbox)
        assert obj.object_id == "id1"
        assert obj.class_name == "disk_1"
        assert obj.confidence == pytest.approx(0.95)
        np.testing.assert_array_equal(obj.position_m, pos)
