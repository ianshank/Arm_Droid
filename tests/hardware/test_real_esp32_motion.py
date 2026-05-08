"""Real-hardware motion + e-stop tests for the ESP32-JSON driver.

Sends small-amplitude moves and verifies the firmware-reported state
reflects the command. Tests latched e-stop both as a host-issued command
and as a watchdog auto-latch (host falls silent).
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from armdroid.config.schema import ArmSettings
from armdroid.hardware.esp32_json_driver import Esp32JsonDriver
from armdroid.protocols import ArmCommandRejected


@pytest.mark.asyncio
async def test_small_move_reflected_in_state(hil_settings: ArmSettings) -> None:
    """Send a small move, wait for completion, verify state.

    Uses a small amplitude so this test is safe to run with any hobby
    arm pose without hitting joint limits.
    """
    drv = Esp32JsonDriver(hil_settings.arm)
    await drv.connect()
    try:
        target = (0.05,) * hil_settings.arm.dof
        await drv.send_joint_positions(target, duration_s=1.0)
        # Wait for the move to complete
        await asyncio.sleep(1.5)
        state = await drv.read_state()
        # The firmware reports the commanded pose with millisecond ts;
        # values within 0.05 rad are well inside servo precision.
        for actual, commanded in zip(state.joint_positions, target, strict=True):
            assert actual == pytest.approx(commanded, abs=0.1)
    finally:
        # Return arm to home before disconnecting
        try:
            await drv.home()
        finally:
            await drv.disconnect()


@pytest.mark.asyncio
async def test_estop_blocks_motion(hil_settings: ArmSettings) -> None:
    """Latched e-stop blocks subsequent set_joints; clear restores motion."""
    drv = Esp32JsonDriver(hil_settings.arm)
    await drv.connect()
    try:
        await drv.emergency_stop()
        await asyncio.sleep(0.1)  # let firmware ack
        with pytest.raises(ArmCommandRejected):
            await drv.send_joint_positions(
                (0.0,) * hil_settings.arm.dof,
                duration_s=1.0,
            )
        await drv.clear_emergency_stop()
        # Now a motion command should succeed
        await drv.send_joint_positions(
            (0.0,) * hil_settings.arm.dof,
            duration_s=1.0,
        )
    finally:
        await drv.disconnect()


@pytest.mark.asyncio
async def test_watchdog_auto_latches_when_host_silent(
    hil_settings: ArmSettings,
) -> None:
    """If the host stops sending commands, firmware auto-latches e-stop.

    This test waits for `watchdog_timeout_s + buffer` and verifies the
    next `read_state` shows `estop_active=True`. The driver's keepalive
    loop is bypassed by directly accessing `_latest_state` (and not
    issuing any new commands) for the duration of the timeout.
    """
    drv = Esp32JsonDriver(hil_settings.arm)
    await drv.connect()
    try:
        # Cancel the keepalive loop so it does not issue pings during the
        # watchdog silence window — the test deliberately holds the wire
        # idle to trigger the firmware auto-latch.
        if drv._keepalive_task is not None:  # type: ignore[attr-defined]
            drv._keepalive_task.cancel()  # type: ignore[attr-defined]
            with contextlib.suppress(asyncio.CancelledError):
                await drv._keepalive_task  # type: ignore[attr-defined]
        # Wait longer than the watchdog timeout — firmware should latch
        wait_s = hil_settings.arm.firmware.watchdog_timeout_s + 0.5
        await asyncio.sleep(wait_s)
        state = await drv.read_state()
        assert state.estop_active
    finally:
        # Clear and disconnect cleanly
        try:
            await drv.clear_emergency_stop()
        finally:
            await drv.disconnect()
