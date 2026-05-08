"""Smoke tests for ``armdroid.domain.units`` typed scalar wrappers."""

from __future__ import annotations

from armdroid.domain import units


def test_unit_aliases_are_callable_and_transparent() -> None:
    """``NewType`` aliases must wrap floats without runtime overhead."""
    rad = units.Radians(1.5)
    sec = units.Seconds(0.1)
    hz = units.Hz(50.0)
    n = units.Newtons(2.0)
    m = units.Meters(0.25)

    assert rad == 1.5
    assert sec == 0.1
    assert hz == 50.0
    assert n == 2.0
    assert m == 0.25


def test_unit_module_public_surface() -> None:
    """The module documents its public API via ``__all__``."""
    assert set(units.__all__) == {"Hz", "Meters", "Newtons", "Radians", "Seconds"}
    for name in units.__all__:
        assert hasattr(units, name)
