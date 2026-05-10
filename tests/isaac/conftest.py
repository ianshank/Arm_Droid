"""Isaac Sim smoke-test conftest (PR-B B.15).

Mirrors ``tests/hardware/conftest.py`` for the Isaac Sim suite. Tests
under ``tests/isaac/`` only run when:

1. ``ARMDROID_ISAAC_RUN=1`` env var is set (explicit opt-in — default
   ``pytest tests/`` runs do NOT attempt to boot Kit on a developer
   machine that just happens to have isaaclab installed).
2. ``isaaclab`` is importable (``pip install -e ".[isaac]" --extra-index-url
   https://pypi.nvidia.com``).

The ``pytest_collection_modifyitems`` hook auto-marks every collected
test in this directory with ``@pytest.mark.isaac`` AND
``@pytest.mark.gpu`` so ``make test`` (which excludes both markers by
default) skips them cleanly.

Session-scoped Kit fixture
--------------------------
``isaac_session_driver`` boots Kit + the SO-ARM articulation **once**
per pytest invocation and yields the connected driver. All smoke tests
share this instance — Isaac Sim's ``AppLauncher`` is a process-wide
singleton (``_app_state``), so each test creating its own driver would
hit a double-launch error after the first ``connect()``. PR-11 review
fix: copilot ``H-disconnect-doesnt-clear-flag`` /
``H-smoke-tests-singleton-fail``.

CI runs locally only — the GPU-bound smoke suite cannot run on
``ubuntu-latest`` without a CUDA GPU. The ``gpu-ci.yml`` workflow
(B.18) runs the pure-Python isaac-extra unit tests under
``tests/unit/hardware/isaac_sim/`` etc., not these.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

if TYPE_CHECKING:
    from armdroid.hardware.isaac_sim.driver import IsaacSimDriver


def _try_isaac_available() -> bool:
    """Return True iff ARMDROID_ISAAC_RUN=1 AND isaaclab is importable."""
    if os.environ.get("ARMDROID_ISAAC_RUN") != "1":
        return False
    try:
        import isaaclab  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="session")
def isaac_available() -> None:
    """Skip cleanly when Isaac Sim is unavailable.

    Yields ``None`` rather than a value because the caller only cares
    about gating, not about an isaaclab handle (the actual handle is
    held by ``IsaacSimDriver`` / ``SoArmReachIsaacEnv``).
    """
    if not _try_isaac_available():
        pytest.skip(
            "Isaac Sim/Lab not available (need ARMDROID_ISAAC_RUN=1 and "
            "the [isaac] extra installed via "
            'pip install -e ".[isaac]" --extra-index-url https://pypi.nvidia.com)'
        )


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def isaac_session_driver(
    isaac_available: None,
) -> AsyncIterator[IsaacSimDriver]:
    """Connect a single ``IsaacSimDriver`` for the entire pytest session.

    Boots Kit + SimulationContext + the SO-ARM articulation exactly
    once. All smoke tests in this directory should depend on this
    fixture rather than calling ``IsaacSimDriver(...).connect()``
    directly — Kit cannot be re-launched after teardown in the same
    process, so per-test boot/shutdown is unsafe.
    """
    from armdroid.config.schema import ArmSettings
    from armdroid.hardware.isaac_sim.driver import IsaacSimDriver

    cfg = ArmSettings(arm_driver_kind="isaac_sim")
    drv = IsaacSimDriver(cfg.arm, sim_cfg=cfg.arm_sim_isaac)
    await drv.connect()
    try:
        yield drv
    finally:
        await drv.disconnect()


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Auto-mark every test in this directory with ``isaac`` AND ``gpu``."""
    for item in items:
        if "tests/isaac" in str(item.fspath).replace("\\", "/"):
            item.add_marker(pytest.mark.isaac)
            item.add_marker(pytest.mark.gpu)
