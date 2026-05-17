"""Regression: ``IsaacSimDriver.connect`` must boot Kit on the main thread.

``isaacsim.simulation_app.SimulationApp.__init__`` registers a
``SIGINT`` handler via :func:`signal.signal`, which Python only allows
from the main thread of the main interpreter. The pre-fix ``connect``
wrapped ``_launch_kit_and_articulation`` in ``asyncio.to_thread(...)``,
which schedules work on the default ``ThreadPoolExecutor`` and trips
``ValueError: signal only works in main thread of the main interpreter``
on the very first ``await drv.connect()``.

These tests do not require the ``[isaac]`` extra — they exercise the
control flow in ``connect`` while patching the heavy work, so they
run in the default CI matrix.
"""

from __future__ import annotations

import asyncio
import threading
import types
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from armdroid.config.schema import ArmConfig
from armdroid.config.schema.sim_isaac import _default_sim_cfg
from armdroid.hardware.isaac_sim import _app_state
from armdroid.hardware.isaac_sim.driver import IsaacSimDriver

if TYPE_CHECKING:
    from collections.abc import Iterator


def _stub_module(name: str) -> types.ModuleType:
    """Bare ModuleType so the ``isaaclab.app`` probe import succeeds."""
    return types.ModuleType(name)


@pytest.fixture(autouse=True)
def _reset_app_launcher_singleton() -> Iterator[None]:
    """Reset the AppLauncher process-singleton between tests."""
    _app_state.reset_for_tests()
    yield
    _app_state.reset_for_tests()


@pytest.mark.asyncio
async def test_connect_runs_launch_on_main_thread() -> None:
    """``_launch_kit_and_articulation`` must execute on the main thread.

    Regression for the ``ValueError: signal only works in main thread``
    crash when ``connect`` wrapped the launch in ``asyncio.to_thread``.
    """
    cfg = ArmConfig()
    drv = IsaacSimDriver(cfg, sim_cfg=_default_sim_cfg())

    main_thread_id = threading.main_thread().ident
    captured: dict[str, int | None] = {"thread_id": None}

    def _stub_launch() -> None:
        captured["thread_id"] = threading.current_thread().ident

    with (
        patch.object(drv, "_launch_kit_and_articulation", side_effect=_stub_launch),
        # Probe import inside connect() succeeds when isaaclab.app is importable;
        # patch it in case the test runs without the [isaac] extra.
        patch.dict(
            "sys.modules",
            {
                "isaaclab": _stub_module("isaaclab"),
                "isaaclab.app": _stub_module("isaaclab.app"),
            },
        ),
    ):
        await drv.connect()

    assert captured["thread_id"] == main_thread_id, (
        "Launch ran on a worker thread instead of the main thread; "
        "Isaac Sim's signal.signal call would fail."
    )
    assert drv.is_connected is True


@pytest.mark.asyncio
async def test_connect_does_not_use_to_thread_for_launch() -> None:
    """``connect`` must not dispatch the launch via ``asyncio.to_thread``.

    Belt-and-braces guard: the source-of-truth for the main-thread
    requirement is the ``signal.signal`` call inside Isaac Sim's
    ``SimulationApp``. We assert here at the asyncio-API surface that
    ``to_thread`` is never invoked during ``connect``, so a future
    refactor cannot silently reintroduce the regression.
    """
    cfg = ArmConfig()
    drv = IsaacSimDriver(cfg, sim_cfg=_default_sim_cfg())

    with (
        patch.object(drv, "_launch_kit_and_articulation", return_value=None),
        patch.dict(
            "sys.modules",
            {
                "isaaclab": _stub_module("isaaclab"),
                "isaaclab.app": _stub_module("isaaclab.app"),
            },
        ),
        patch("armdroid.hardware.isaac_sim.driver.asyncio.to_thread") as mock_to_thread,
    ):
        await drv.connect()

    assert not mock_to_thread.called, (
        "connect() called asyncio.to_thread; the bootstrap must run on "
        "the main thread to satisfy SimulationApp's signal.signal "
        "registration. See docstring on _launch_kit_and_articulation."
    )


@pytest.mark.asyncio
async def test_connect_idempotent() -> None:
    """Second ``connect`` call is a no-op (existing contract; protect alongside fix)."""
    cfg = ArmConfig()
    drv = IsaacSimDriver(cfg, sim_cfg=_default_sim_cfg())

    call_count = {"n": 0}

    def _count() -> None:
        call_count["n"] += 1

    with (
        patch.object(drv, "_launch_kit_and_articulation", side_effect=_count),
        patch.dict(
            "sys.modules",
            {
                "isaaclab": _stub_module("isaaclab"),
                "isaaclab.app": _stub_module("isaaclab.app"),
            },
        ),
    ):
        await drv.connect()
        await drv.connect()

    assert call_count["n"] == 1
    assert drv.is_connected is True


def test_connect_stays_async() -> None:
    """``connect`` itself must remain ``async`` so callers can await it."""
    assert asyncio.iscoroutinefunction(IsaacSimDriver.connect), (
        "connect() must be async — synchronous bootstrap inside an "
        "async method is the design (event loop blocks for ~15 s on "
        "first call); changing the method to sync would break every "
        "caller."
    )
