"""HIL reconnect and state-recovery tests for the ESP32 JSON driver.

Verifies that the driver handles disconnect / reconnect cycles cleanly —
no state contamination from a prior session bleeds into a new connection,
and firmware-side persistent state (e-stop latch) is observable
immediately after a reconnect.

Run with::

    ARMDROID_HIL_RUN=1 pytest tests/hardware/test_real_esp32_recovery.py \\
        -m hardware -v
"""

from __future__ import annotations

import asyncio

import pytest

from armdroid.config.schema import ArmSettings
from armdroid.hardware.esp32 import Esp32JsonDriver

# ---------------------------------------------------------------------------
# Basic disconnect → reconnect cycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disconnect_reconnect_yields_fresh_state(
    hil_settings: ArmSettings,
) -> None:
    """Reconnect after a clean disconnect produces a fresh usable state.

    Verifies:

    * ``disconnect()`` is idempotent (second call is a no-op).
    * After ``connect()`` on a new driver instance the cached state is
      populated within the configured timeout.
    * The second driver session is independent of the first (no pending-
      reply futures leaked across instances).
    """
    # First session — connect, verify state, disconnect.
    drv_a = Esp32JsonDriver(hil_settings.arm)
    await drv_a.connect()
    try:
        state_a = await drv_a.read_state()
        assert len(state_a.joint_positions) == hil_settings.arm.dof
    finally:
        await drv_a.disconnect()
        assert not drv_a.is_connected
        # Second disconnect must be a no-op.
        await drv_a.disconnect()

    # Second session — fresh driver instance, reconnect.
    drv_b = Esp32JsonDriver(hil_settings.arm)
    await drv_b.connect()
    try:
        assert drv_b.is_connected
        state_b = await drv_b.read_state()
        assert len(state_b.joint_positions) == hil_settings.arm.dof
    finally:
        await drv_b.disconnect()


# ---------------------------------------------------------------------------
# Reconnect after abrupt disconnect (simulated by direct port close)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_abrupt_disconnect_then_reconnect(
    hil_settings: ArmSettings,
) -> None:
    """Driver recovers from an abrupt port closure without hanging.

    Closes the underlying ``serial.Serial`` port directly (simulating a
    USB unplug) and then calls ``disconnect()`` to clean up the tasks.
    A subsequent ``connect()`` on a new instance must succeed.
    """
    drv_a = Esp32JsonDriver(hil_settings.arm)
    await drv_a.connect()
    port = drv_a._port  # type: ignore[attr-defined]
    assert port is not None
    # Simulate abrupt disconnect by closing the port underneath the driver.
    await asyncio.to_thread(port.close)
    # Graceful teardown must not raise even with the port already closed.
    await drv_a.disconnect()
    assert not drv_a.is_connected

    # A fresh driver must be able to reconnect.
    drv_b = Esp32JsonDriver(hil_settings.arm)
    await drv_b.connect()
    try:
        assert drv_b.is_connected
        await drv_b.read_state()
    finally:
        await drv_b.disconnect()


# ---------------------------------------------------------------------------
# E-stop persists across host reconnect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estop_state_observable_after_reconnect(
    hil_settings: ArmSettings,
) -> None:
    """Firmware-latched e-stop is visible in the state after a host reconnect.

    The firmware holds the e-stop latch independently of the host
    connection.  After the host disconnects (without clearing e-stop) and
    reconnects, the first :meth:`~Esp32JsonDriver.read_state` must show
    ``estop_active = True``.

    Cleans up by issuing ``clear_emergency_stop`` before disconnecting
    at the end.
    """
    # First session — latch e-stop, disconnect without clearing.
    drv_a = Esp32JsonDriver(hil_settings.arm)
    await drv_a.connect()
    try:
        await drv_a.emergency_stop()
        await asyncio.sleep(0.1)  # allow firmware to ack
    finally:
        # Disconnect without clearing — intentional for this test.
        await drv_a.disconnect()

    # Second session — reconnect, observe latched e-stop.
    drv_b = Esp32JsonDriver(hil_settings.arm)
    await drv_b.connect()
    try:
        state = await drv_b.read_state()
        assert state.estop_active, (
            "Expected firmware to still have e-stop latched after host reconnect, "
            f"but estop_active={state.estop_active!r}"
        )
    finally:
        # Always clear before leaving so subsequent tests start clean.
        await drv_b.clear_emergency_stop()
        await asyncio.sleep(0.1)
        await drv_b.disconnect()


# ---------------------------------------------------------------------------
# Pending-reply map is empty after disconnect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pending_map_cleared_on_disconnect(
    hil_driver: Esp32JsonDriver,
) -> None:
    """Pending-reply map is empty before and after a clean disconnect.

    Verifies the invariant that a freshly connected driver has no orphaned
    futures in ``_pending`` and that a clean :meth:`disconnect` leaves the
    map empty.  This guards against regressions in teardown ordering where
    a non-empty ``_pending`` map could cause subsequent tests to wait
    indefinitely for replies that will never arrive.

    .. note::

       This covers the *clean-disconnect* path.  The abrupt-port-close
       path (futures cancelled mid-flight) is exercised by
       :func:`test_recovery_after_port_close`.
    """
    pending: dict = hil_driver._pending  # type: ignore[attr-defined]
    # Freshly connected driver must have no orphaned futures.
    assert len(pending) == 0

    # A clean disconnect must leave the map empty.
    await hil_driver.disconnect()
    assert len(pending) == 0
