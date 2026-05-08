"""Tests for the generic :class:`armdroid._registry.Registry` plugin lookup."""

from __future__ import annotations

import pytest

from armdroid._registry import Registry, RegistryError


class _Foo:
    pass


class _Bar:
    pass


def test_register_and_get_roundtrip() -> None:
    reg: Registry[type] = Registry("widget")
    reg.register("foo", _Foo)
    assert reg.get("foo") is _Foo
    assert "foo" in reg
    assert reg.available() == ["foo"]


def test_re_register_same_factory_is_idempotent() -> None:
    reg: Registry[type] = Registry("widget")
    reg.register("foo", _Foo)
    reg.register("foo", _Foo)  # no error
    assert reg.get("foo") is _Foo


def test_re_register_different_factory_raises() -> None:
    reg: Registry[type] = Registry("widget")
    reg.register("foo", _Foo)
    with pytest.raises(RegistryError, match="already registered"):
        reg.register("foo", _Bar)


def test_override_replaces_registration() -> None:
    reg: Registry[type] = Registry("widget")
    reg.register("foo", _Foo)
    reg.override("foo", _Bar)
    assert reg.get("foo") is _Bar


def test_unregister_is_noop_for_missing_keys() -> None:
    reg: Registry[type] = Registry("widget")
    reg.unregister("nope")  # must not raise


def test_get_unknown_lists_available_names() -> None:
    reg: Registry[type] = Registry("widget")
    reg.register("foo", _Foo)
    reg.register("bar", _Bar)
    with pytest.raises(RegistryError, match=r"unknown widget 'baz'.*bar, foo"):
        reg.get("baz")


def test_membership_only_matches_strings() -> None:
    reg: Registry[type] = Registry("widget")
    reg.register("foo", _Foo)
    assert "foo" in reg
    assert 1 not in reg  # type: ignore[operator]


def test_load_entry_points_without_group_returns_zero() -> None:
    reg: Registry[type] = Registry("widget")
    assert reg.load_entry_points() == 0


def test_load_entry_points_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    reg: Registry[type] = Registry("widget", entry_point_group="armdroid._test_none")

    calls = {"n": 0}

    def fake_entry_points(group: str) -> list[object]:
        del group
        calls["n"] += 1
        return []

    monkeypatch.setattr("armdroid._registry.metadata.entry_points", fake_entry_points)
    assert reg.load_entry_points() == 0
    assert reg.load_entry_points() == 0  # second call short-circuits
    assert calls["n"] == 1


def test_load_entry_points_collects_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeEP:
        def __init__(self, name: str, target: type) -> None:
            self.name = name
            self._target = target

        def load(self) -> type:
            return self._target

    eps = [_FakeEP("foo", _Foo), _FakeEP("bar", _Bar)]
    monkeypatch.setattr(
        "armdroid._registry.metadata.entry_points",
        lambda group: eps if group == "armdroid._test" else [],
    )

    reg: Registry[type] = Registry("widget", entry_point_group="armdroid._test")
    loaded = reg.load_entry_points()
    assert loaded == 2
    assert reg.get("foo") is _Foo
    assert reg.get("bar") is _Bar


def test_load_entry_points_continues_past_broken_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _GoodEP:
        name = "good"

        def load(self) -> type:
            return _Foo

    class _BadEP:
        name = "bad"

        def load(self) -> type:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "armdroid._registry.metadata.entry_points",
        lambda group: [_BadEP(), _GoodEP()] if group == "armdroid._test" else [],
    )
    reg: Registry[type] = Registry("widget", entry_point_group="armdroid._test")
    assert reg.load_entry_points() == 1
    assert "good" in reg
    assert "bad" not in reg
