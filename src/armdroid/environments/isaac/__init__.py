"""Isaac Lab 2.3 environment wrappers (PR-B B.11a).

Re-exports :class:`SoArmReachIsaacEnv`. Module top-level deliberately
does NOT import isaaclab — the env wrapper does that lazily inside
``_ensure_built()``.

Coverage-omit: ``armdroid.environments.isaac.{__init__,reach}.py`` are
in ``[tool.coverage.run].omit`` (per PR-A). Tests live under
``tests/isaac/`` and only run with ``ARMDROID_ISAAC_RUN=1`` + a CUDA GPU.
"""

from __future__ import annotations

from armdroid.environments.isaac.reach import SoArmReachIsaacEnv

__all__ = ["SoArmReachIsaacEnv"]
