"""YAML overlay loader for armdroid configuration.

Loads optional default.yaml from the repo's ``config/`` directory, then
merges environment-specific overlays. Environment variables with the
``ARMDROID_`` prefix override all (handled by pydantic-settings on the
Settings class itself).

Vendored from mousedroid with the env prefix changed to ``ARMDROID_`` and
the loader factored into pure dict helpers + a generic ``load_settings``
that takes any pydantic-settings ``BaseSettings`` subclass.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import yaml

from armdroid.logging.setup import get_logger

if TYPE_CHECKING:
    from pydantic_settings import BaseSettings

_log = get_logger(__name__)

# src/armdroid/config/loader.py -> ../../../../config/
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "config"

_ENV_PREFIX = "ARMDROID_"

T = TypeVar("T", bound="BaseSettings")


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` (in place).

    Args:
        base: Base configuration dictionary (mutated).
        overlay: Overlay values to merge on top.

    Returns:
        The merged ``base`` dictionary.
    """
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a single YAML file and return its contents as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML contents (empty dict if file is empty).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with path.open() as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    return data


def merge_yaml_overlays(
    *overlay_paths: Path,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    """Merge default.yaml + overlay YAML files into a single dict.

    Merge order:
        1. ``<config_dir>/default.yaml`` (if present)
        2. Each ``overlay_path`` in order

    Args:
        overlay_paths: Additional YAML files to merge on top of defaults.
        config_dir: Directory containing ``default.yaml``. Defaults to repo ``config/``.

    Returns:
        Merged configuration dictionary.
    """
    base_dir = config_dir or _DEFAULT_CONFIG_DIR
    default_path = base_dir / "default.yaml"

    merged: dict[str, Any] = {}
    if default_path.exists():
        merged = load_yaml(default_path)
        _log.debug("config_base_loaded", path=str(default_path))
    else:
        _log.debug("config_no_default_yaml", path=str(default_path))

    for overlay_path in overlay_paths:
        overlay_data = load_yaml(overlay_path)
        _deep_merge(merged, overlay_data)
        _log.debug("config_overlay_applied", path=str(overlay_path))

    _log.info("config_settings_resolved", n_overlays=len(overlay_paths))
    return merged


def load_settings(
    *overlay_paths: Path,
    settings_class: type[T],
    config_dir: Path | None = None,
) -> T:
    """Load a Settings instance by merging YAML overlays + env vars.

    Merge order:
        1. ``<config_dir>/default.yaml`` (base)
        2. Each ``overlay_path`` in order
        3. Environment variables with the ``ARMDROID_`` prefix
           (handled internally by pydantic-settings)

    Top-level fields that are also set via ``ARMDROID_<FIELD>`` env vars
    are removed from the merged dict so the env var source wins. Empty
    nested env vars (e.g. ``ARMDROID_SECTION__FIELD=""``) are temporarily
    unset to avoid pydantic-settings materializing empty optional configs.

    Args:
        overlay_paths: Additional YAML files to merge on top of defaults.
        settings_class: pydantic-settings ``BaseSettings`` subclass to construct.
        config_dir: Directory containing ``default.yaml``.

    Returns:
        Instance of ``settings_class``.
    """
    merged = merge_yaml_overlays(*overlay_paths, config_dir=config_dir)

    # Drop top-level keys overridden by ARMDROID_<KEY> env vars so the env
    # var source wins (pydantic-settings v2 gives init kwargs higher priority
    # than env vars).
    env_overridden = {
        k[len(_ENV_PREFIX) :].lower()
        for k in os.environ
        if k.upper().startswith(_ENV_PREFIX) and "__" not in k[len(_ENV_PREFIX) :]
    }
    for key in env_overridden:
        merged.pop(key, None)

    # Drop empty nested env vars (e.g. ARMDROID_SECTION__FIELD="") which
    # would otherwise materialize an empty optional config and fail validation.
    empty_nested = [
        k
        for k, v in os.environ.items()
        if k.upper().startswith(_ENV_PREFIX)
        and "__" in k[len(_ENV_PREFIX) :]
        and not v.strip()
    ]
    with _ScopedEnvUnset(empty_nested):
        return settings_class(**merged)


class _ScopedEnvUnset:
    """Context manager that temporarily removes the given env vars."""

    def __init__(self, names: list[str]) -> None:
        """Initialise the unset scope.

        Args:
            names: Environment variable names to remove on enter.
        """
        self._names = names
        self._saved: dict[str, str] = {}

    def __enter__(self) -> _ScopedEnvUnset:
        """Remove the configured env vars and remember their values."""
        for name in self._names:
            if name in os.environ:
                self._saved[name] = os.environ.pop(name)
                _log.debug("config_env_empty_nested_dropped", name=name)
        return self

    def __exit__(self, *_exc: object) -> None:
        """Restore previously removed env vars."""
        for name, value in self._saved.items():
            os.environ[name] = value
