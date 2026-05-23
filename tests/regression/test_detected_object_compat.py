"""Regression: ``DetectedObject`` ctor preserves the v0.2 surface.

Pins the legacy six-arg positional constructor and the exact ``__slots__``
tuple so the Phase A widening does not silently drift away from
backwards-compatibility guarantees made in ``CLAUDE.md`` and the
peer-reviewed Phase A plan.
"""

from __future__ import annotations

import numpy as np

from armdroid.domain.state import DetectedObject

_EXPECTED_SLOTS: tuple[str, ...] = (
    "affordances",
    "bbox",
    "class_name",
    "confidence",
    "is_fixed",
    "is_fragile",
    "object_id",
    "orientation_rad",
    "position_m",
    "semantic_tags",
    "text_query",
)


def _legacy_kwargs() -> dict[str, object]:
    return {
        "object_id": "id1",
        "class_name": "disk_1",
        "confidence": 0.95,
        "position_m": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        "orientation_rad": np.zeros(3, dtype=np.float64),
        "bbox": np.array([0, 0, 10, 10], dtype=np.float64),
    }


def test_positional_ctor_six_args_still_works() -> None:
    """The v0.2 positional ctor must remain identical so every existing
    call site (``perception.object_detector``, integration + unit tests)
    keeps working without modification.
    """
    obj = DetectedObject(
        "id1",
        "disk_1",
        0.95,
        np.array([0.1, 0.2, 0.3], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.array([0, 0, 10, 10], dtype=np.float64),
    )
    assert obj.object_id == "id1"
    assert obj.class_name == "disk_1"
    assert obj.confidence == 0.95


def test_new_fields_default_falsy() -> None:
    obj = DetectedObject(**_legacy_kwargs())
    assert obj.affordances == ()
    assert obj.is_fragile is False
    assert obj.is_fixed is False
    assert obj.semantic_tags == ()
    assert obj.text_query is None


def test_open_vocab_kwargs_accepted() -> None:
    obj = DetectedObject(
        **_legacy_kwargs(),
        affordances=("graspable", "stackable"),
        is_fragile=True,
        is_fixed=False,
        semantic_tags=("tool", "metal"),
        text_query="the small wrench",
    )
    assert obj.affordances == ("graspable", "stackable")
    assert obj.is_fragile is True
    assert obj.semantic_tags == ("tool", "metal")
    assert obj.text_query == "the small wrench"


def test_slots_pinned_to_expected_tuple() -> None:
    """``__slots__`` drift will break wheel size and pickle compatibility.

    The tuple is asserted as a set because slot declaration order is not
    contractual; only the membership is.
    """
    assert set(DetectedObject.__slots__) == set(_EXPECTED_SLOTS)


def test_new_fields_rejected_as_positional() -> None:
    """The new fields are keyword-only - using them positionally would
    silently consume legacy positional slots if not guarded.
    """
    try:
        DetectedObject(
            "id1",
            "disk_1",
            0.95,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            ("graspable",),  # type: ignore[misc]
        )
    except TypeError:
        return
    raise AssertionError("Expected TypeError for positional new-field usage")
