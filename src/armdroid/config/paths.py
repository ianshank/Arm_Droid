"""Single helper for resolving filesystem paths to vendored assets.

Used by both MuJoCo and Isaac Sim path consumers — the only place that
walks the repo layout. Keeps every other module free of CWD-relative
``Path("assets/...")`` constructs that would break under different
working directories or installed-package layouts.

Resolution order for :func:`resolve_asset_path`:

1. ``base`` argument when given (explicit wins).
2. ``$ARMDROID_REPO_ROOT`` env var.
3. Walk parents from this module's ``__file__`` looking for
   ``pyproject.toml``.

Resolution does **not** check existence — callers branch on
``Path.is_file()`` so that mock/optional asset paths still construct.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

REPO_ROOT_ENV: Final[str] = "ARMDROID_REPO_ROOT"
SO101_ASSETS_REL: Final[Path] = Path("assets/so_arm/so101")
SO101_URDF_REL: Final[Path] = SO101_ASSETS_REL / "urdf" / "so101_new_calib.urdf"
SO101_MJCF_SCENE_REL: Final[Path] = SO101_ASSETS_REL / "mjcf" / "scene.xml"
SO101_USD_REL: Final[Path] = SO101_ASSETS_REL / "usd" / "so101.usd"


def _find_repo_root_from(start: Path) -> Path:
    """Walk parents from ``start`` looking for ``pyproject.toml``.

    Falls back to ``start`` if no marker found within the parent chain.

    Args:
        start: Starting directory (typically ``Path(__file__).resolve()``).

    Returns:
        Repo root if found, else ``start``.
    """
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


def resolve_asset_path(
    rel: str | Path,
    *,
    base: Path | None = None,
) -> Path:
    """Resolve a repo-relative asset path to absolute.

    Args:
        rel: Relative path (or absolute, in which case returned as-is).
        base: Optional explicit base. Overrides the env var and the
            pyproject walk when given.

    Returns:
        Absolute path. **Does not** check existence — callers do that.
    """
    p = Path(rel)
    if p.is_absolute():
        return p.resolve()
    if base is not None:
        return (base / p).resolve()
    env_base = os.environ.get(REPO_ROOT_ENV)
    if env_base:
        return (Path(env_base) / p).resolve()
    return (_find_repo_root_from(Path(__file__).resolve()) / p).resolve()


def resolve_so101_urdf(*, base: Path | None = None) -> Path:
    """Return the absolute path to the vendored SO-ARM101 URDF."""
    return resolve_asset_path(SO101_URDF_REL, base=base)


def resolve_so101_mjcf_scene(*, base: Path | None = None) -> Path:
    """Return the absolute path to the vendored SO-ARM101 MJCF scene."""
    return resolve_asset_path(SO101_MJCF_SCENE_REL, base=base)


def resolve_so101_usd(*, base: Path | None = None) -> Path:
    """Return the absolute path to the SO-ARM101 USD (a build artefact)."""
    return resolve_asset_path(SO101_USD_REL, base=base)


__all__ = [
    "REPO_ROOT_ENV",
    "SO101_ASSETS_REL",
    "SO101_MJCF_SCENE_REL",
    "SO101_URDF_REL",
    "SO101_USD_REL",
    "resolve_asset_path",
    "resolve_so101_mjcf_scene",
    "resolve_so101_urdf",
    "resolve_so101_usd",
]
