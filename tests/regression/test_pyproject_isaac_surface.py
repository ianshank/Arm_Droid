"""Regression: pyproject.toml declares [isaac] extras + NVIDIA index docs.

Closes PR-B B.1: ``[project.optional-dependencies].isaac`` pins Isaac Lab
2.3 and RSL-RL ``>=2.0``. Without this regression, a developer can pin
isaaclab without documenting the NVIDIA index URL — a common failure
mode that makes ``pip install -e ".[isaac]"`` fail with "no matching
distribution" because PyPI doesn't host ``isaacsim``.

Entry-point membership is intentionally NOT asserted here — that surface
is owned by ``tests/regression/test_entry_point_mirror.py`` which
walks every plugin group + asserts in-code → pyproject mirror. This
file owns only the extras-shape contract.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_README = _REPO_ROOT / "README.md"
_GPU_CI = _REPO_ROOT / ".github" / "workflows" / "gpu-ci.yml"

#: Pattern accepts ``2.3.0`` or any ``2.3.X`` patch (PEP 440 micro upgrades).
_ISAACLAB_VERSION_RE = re.compile(r"isaaclab\[.*\]==2\.3\.\d+")
_RSL_RL_VERSION_RE = re.compile(r"rsl-rl-lib>=2\.\d+(\.\d+)?")
_NVIDIA_INDEX = "pypi.nvidia.com"


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    with _PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


class TestIsaacOptionalExtra:
    def test_isaac_extra_present(self, pyproject: dict[str, Any]) -> None:
        extras = pyproject["project"]["optional-dependencies"]
        assert "isaac" in extras, (
            "PR-B requires [project.optional-dependencies.isaac] to pin "
            "Isaac Lab + RSL-RL. Mirrored in ADR-0005."
        )

    def test_isaac_extra_pins_isaaclab_2_3(self, pyproject: dict[str, Any]) -> None:
        deps = pyproject["project"]["optional-dependencies"]["isaac"]
        assert any(
            _ISAACLAB_VERSION_RE.match(d) for d in deps
        ), f"[isaac] must pin isaaclab[...]==2.3.X. Got: {deps!r}"

    def test_isaac_extra_pins_rsl_rl(self, pyproject: dict[str, Any]) -> None:
        deps = pyproject["project"]["optional-dependencies"]["isaac"]
        assert any(
            _RSL_RL_VERSION_RE.match(d) for d in deps
        ), f"[isaac] must pin rsl-rl-lib>=2.X. Got: {deps!r}"

    def test_isaac_extra_does_not_pin_torch(
        self,
        pyproject: dict[str, Any],
    ) -> None:
        """Closes R2 — speculative torch / torchvision pins are forbidden.

        torch is in base [project.dependencies]; isaaclab's transitive
        resolution handles its own torch needs. Pinning torch here would
        risk version conflicts.
        """
        deps = pyproject["project"]["optional-dependencies"]["isaac"]
        assert not any(
            d.startswith("torch") for d in deps
        ), f"[isaac] must NOT pin torch / torchvision. Got: {deps!r}"


class TestNvidiaIndexUrlDocumented:
    """The NVIDIA index URL is required for `isaaclab` to resolve.

    Without it, ``pip install -e ".[isaac]"`` fails with "no matching
    distribution found for isaacsim" because PyPI does not host the
    package — it lives on ``https://pypi.nvidia.com``. PR-B B.1 must
    document this in the README install section AND the gpu-ci
    workflow MUST set it via ``PIP_EXTRA_INDEX_URL`` so CI can install
    the extra.
    """

    def test_readme_documents_nvidia_index(self) -> None:
        text = _README.read_text(encoding="utf-8")
        assert _NVIDIA_INDEX in text, (
            f"README.md must mention {_NVIDIA_INDEX} so contributors "
            "know how to install the [isaac] extra."
        )

    def test_gpu_ci_sets_pip_extra_index_url(self) -> None:
        if not _GPU_CI.is_file():
            pytest.skip(f"{_GPU_CI} not yet created (lands in B.18)")
        text = _GPU_CI.read_text(encoding="utf-8")
        assert _NVIDIA_INDEX in text, (
            f".github/workflows/gpu-ci.yml must set the NVIDIA index "
            f"({_NVIDIA_INDEX}) so the workflow can install [isaac] extras."
        )
