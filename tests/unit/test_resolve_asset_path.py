"""Unit tests for armdroid.config.paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from armdroid.config.paths import (
    REPO_ROOT_ENV,
    SO101_MJCF_SCENE_REL,
    SO101_URDF_REL,
    SO101_USD_REL,
    resolve_asset_path,
    resolve_so101_mjcf_scene,
    resolve_so101_urdf,
    resolve_so101_usd,
)


class TestResolveAssetPath:
    def test_returns_absolute_path_for_relative_input(self, tmp_path: Path) -> None:
        result = resolve_asset_path("foo/bar.urdf", base=tmp_path)
        assert result.is_absolute()
        assert result == (tmp_path / "foo/bar.urdf").resolve()

    def test_passthrough_for_absolute_input(self, tmp_path: Path) -> None:
        absolute = tmp_path / "x.urdf"
        result = resolve_asset_path(absolute, base=tmp_path)
        assert result == absolute.resolve()

    def test_uses_env_var_for_base(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))
        result = resolve_asset_path("a/b.txt")
        assert result == (tmp_path / "a/b.txt").resolve()

    def test_explicit_base_overrides_env_var(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))
        result = resolve_asset_path("c.txt", base=other)
        assert result == (other / "c.txt").resolve()

    def test_walks_to_pyproject_marker_when_no_base_or_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv(REPO_ROOT_ENV, raising=False)
        # The package itself lives inside a repo with pyproject.toml.
        # The walk should find it relative to __file__.
        result = resolve_asset_path("pyproject.toml")
        assert result.is_file()


class TestResolveSo101Helpers:
    def test_resolve_urdf(self, tmp_path: Path) -> None:
        urdf = tmp_path / SO101_URDF_REL
        urdf.parent.mkdir(parents=True)
        urdf.write_text("<robot/>")
        result = resolve_so101_urdf(base=tmp_path)
        assert result.exists()
        assert result.name == "so101_new_calib.urdf"

    def test_resolve_mjcf_scene(self, tmp_path: Path) -> None:
        scene = tmp_path / SO101_MJCF_SCENE_REL
        scene.parent.mkdir(parents=True)
        scene.write_text("<mujoco/>")
        result = resolve_so101_mjcf_scene(base=tmp_path)
        assert result.exists()

    def test_resolve_usd_returns_path_even_if_missing(
        self,
        tmp_path: Path,
    ) -> None:
        # USD is a build artefact — resolver returns the path without
        # checking existence so loaders can branch on .is_file().
        result = resolve_so101_usd(base=tmp_path)
        assert result == (tmp_path / SO101_USD_REL).resolve()
        assert not result.exists()
