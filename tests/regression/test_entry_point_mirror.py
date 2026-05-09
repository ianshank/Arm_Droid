"""Regression: every in-code registry name has a matching pyproject entry point.

The convention documented at ``pyproject.toml`` lines 54-58 is that the
``[project.entry-points."armdroid.<group>"]`` blocks **mirror** the
in-code ``_REGISTRY.register(...)`` calls so that
``importlib.metadata.entry_points("armdroid.<group>")`` reflects the
same surface as ``available_<group>s()``.

This regression test enforces the mirror at runtime so future drift
(adding a registration in code without updating pyproject, or vice
versa) fails CI loudly. The bug it guards against was raised by Devin
+ Copilot on PR #10: ``sac_her`` was added to the in-code registry but
not to the pyproject entry-points block.
"""

from __future__ import annotations

from importlib import metadata

import pytest

from armdroid.environments.registry import available_environments
from armdroid.hardware.registry import available_drivers
from armdroid.planning.registry import available_planners

try:
    from armdroid.perception.registry import available_perception_backends
except ImportError:
    available_perception_backends = None  # type: ignore[assignment]

from armdroid.control.registry import available_rl_agents


def _entry_point_names(group: str) -> set[str]:
    """Return entry-point names declared under ``group`` in installed metadata."""
    eps = metadata.entry_points(group=group)
    return {ep.name for ep in eps}


# ``(group_name, available_callable)`` — every armdroid plugin group + its
# available-names accessor. Adding a new subsystem here automatically
# grows the regression coverage.
_GROUPS: list[tuple[str, object]] = [
    ("armdroid.drivers", available_drivers),
    ("armdroid.environments", available_environments),
    ("armdroid.rl_agents", available_rl_agents),
    ("armdroid.planners", available_planners),
]
if available_perception_backends is not None:
    _GROUPS.append(("armdroid.perception_backends", available_perception_backends))


@pytest.mark.regression
@pytest.mark.parametrize(
    ("group", "available_fn"),
    _GROUPS,
    ids=[g for g, _ in _GROUPS],
)
def test_in_code_registry_mirrors_pyproject_entry_points(
    group: str,
    available_fn: object,
) -> None:
    """Every in-code registration under ``group`` must have a pyproject entry point.

    The reverse direction (entry-point declared but no in-code registration)
    is intentionally NOT asserted because out-of-tree plugins legitimately
    register only via entry points.
    """
    in_code = set(available_fn())  # type: ignore[operator]
    in_pyproject = _entry_point_names(group)

    missing_in_pyproject = in_code - in_pyproject
    assert not missing_in_pyproject, (
        f"In-code registrations under {group!r} are missing matching "
        f"[project.entry-points.{group!r}] entries in pyproject.toml: "
        f"{sorted(missing_in_pyproject)!r}. Add them to keep "
        f"importlib.metadata.entry_points() in sync with available_*()."
    )
