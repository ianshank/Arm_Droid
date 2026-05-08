# ruff: noqa: E402
"""Backwards-compatibility re-export for the legacy ``armdroid.factory`` path.

The canonical home is :mod:`armdroid.orchestration.factory`. New code should
import from :mod:`armdroid` (public façade) or :mod:`armdroid.orchestration`
directly. This shim is preserved for the v0.2.x line and scheduled for
removal in v0.4.0.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from armdroid.factory is deprecated and will be removed in "
    "v0.4.0. Import from armdroid or armdroid.orchestration instead.",
    DeprecationWarning,
    stacklevel=2,
)

from armdroid.orchestration.factory import (
    build_arm_controller,
    build_arm_driver,
    build_arm_environment,
    build_arm_orchestrator,
    build_arm_perception,
    build_arm_planner,
)

__all__ = [
    "build_arm_controller",
    "build_arm_driver",
    "build_arm_environment",
    "build_arm_orchestrator",
    "build_arm_perception",
    "build_arm_planner",
]
