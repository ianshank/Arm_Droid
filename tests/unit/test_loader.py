"""Tests for armdroid.config.loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict

from armdroid.config.loader import (
    load_settings,
    load_yaml,
    merge_yaml_overlays,
)


class _DummySettings(BaseSettings):
    """Minimal settings class for loader tests."""

    model_config = SettingsConfigDict(
        env_prefix="ARMDROID_",
        env_nested_delimiter="__",
    )

    foo: str = "default_foo"
    bar: int = 0
    nested: dict[str, Any] = {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.write_text(yaml.safe_dump(data))


def test_load_yaml_returns_dict(tmp_path: Path) -> None:
    f = tmp_path / "x.yaml"
    _write_yaml(f, {"foo": "bar", "n": 1})
    assert load_yaml(f) == {"foo": "bar", "n": 1}


def test_load_yaml_empty_file_returns_empty_dict(tmp_path: Path) -> None:
    f = tmp_path / "empty.yaml"
    f.write_text("")
    assert load_yaml(f) == {}


def test_merge_yaml_overlays_layers_in_order(tmp_path: Path) -> None:
    base = tmp_path / "default.yaml"
    overlay1 = tmp_path / "o1.yaml"
    overlay2 = tmp_path / "o2.yaml"

    _write_yaml(base, {"foo": "base", "bar": 1, "nested": {"a": 1, "b": 2}})
    _write_yaml(overlay1, {"foo": "from_o1", "nested": {"b": 99}})
    _write_yaml(overlay2, {"bar": 42, "nested": {"c": 3}})

    merged = merge_yaml_overlays(overlay1, overlay2, config_dir=tmp_path)
    assert merged["foo"] == "from_o1"
    assert merged["bar"] == 42
    assert merged["nested"] == {"a": 1, "b": 99, "c": 3}


def test_merge_yaml_overlays_no_default(tmp_path: Path) -> None:
    overlay = tmp_path / "only.yaml"
    _write_yaml(overlay, {"foo": "only"})
    merged = merge_yaml_overlays(overlay, config_dir=tmp_path)
    assert merged == {"foo": "only"}


def test_load_settings_uses_overlays(tmp_path: Path) -> None:
    overlay = tmp_path / "ov.yaml"
    _write_yaml(overlay, {"foo": "from_overlay", "bar": 7})
    cfg = load_settings(
        overlay,
        settings_class=_DummySettings,
        config_dir=tmp_path,
    )
    assert cfg.foo == "from_overlay"
    assert cfg.bar == 7


def test_load_settings_env_var_overrides_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env var ARMDROID_FOO should beat YAML overlay value for top-level field."""
    overlay = tmp_path / "ov.yaml"
    _write_yaml(overlay, {"foo": "from_overlay"})
    monkeypatch.setenv("ARMDROID_FOO", "from_env")
    cfg = load_settings(
        overlay,
        settings_class=_DummySettings,
        config_dir=tmp_path,
    )
    assert cfg.foo == "from_env"


def test_load_settings_drops_empty_nested_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty ARMDROID_SECTION__FIELD env vars are unset before construction."""
    monkeypatch.setenv("ARMDROID_NESTED__SOMETHING", "")
    overlay = tmp_path / "ov.yaml"
    _write_yaml(overlay, {"foo": "ok"})
    cfg = load_settings(
        overlay,
        settings_class=_DummySettings,
        config_dir=tmp_path,
    )
    assert cfg.foo == "ok"
