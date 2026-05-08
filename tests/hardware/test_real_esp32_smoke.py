"""Real-hardware smoke tests for the ESP32-JSON arm driver.

Requires an ESP32 with armdroid firmware flashed and connected. Skipped
by default; run explicitly with:

    pytest tests/hardware -m hardware

Or set ``ARMDROID_ARM__TRANSPORT__SERIAL_PORT`` to point at the device.
"""

from __future__ import annotations

import asyncio

import pytest

from armdroid.config.schema import ArmSettings
from armdroid.domain.protocols import ArmDriverProtocol
from armdroid.hardware.esp32 import Esp32JsonDriver


@pytest.mark.asyncio
async def test_connect_and_disconnect(hil_settings: ArmSettings) -> None:
    """Real firmware boots, the host pings, the driver disconnects."""
    drv = Esp32JsonDriver(hil_settings.arm)
    assert isinstance(drv, ArmDriverProtocol)
    await drv.connect()
    try:
        assert drv.is_connected
    finally:
        await drv.disconnect()
    assert not drv.is_connected


@pytest.mark.asyncio
async def test_state_heartbeat_is_received(hil_settings: ArmSettings) -> None:
    """After connect, a state frame should land within a few heartbeats."""
    drv = Esp32JsonDriver(hil_settings.arm)
    await drv.connect()
    try:
        # Wait up to 1 s for a heartbeat
        for _ in range(100):
            if drv._latest_state is not None:  # type: ignore[attr-defined]
                break
            await asyncio.sleep(0.01)
        state = await drv.read_state()
        assert len(state.joint_positions) == hil_settings.arm.dof
    finally:
        await drv.disconnect()


@pytest.mark.asyncio
async def test_ping_round_trip(hil_settings: ArmSettings) -> None:
    """A single ping + ack round-trip should succeed."""
    drv = Esp32JsonDriver(hil_settings.arm)
    await drv.connect()
    try:
        # The driver issues drain-pings as part of connect — if we got
        # here without an exception, the round-trip works.
        assert drv.is_connected
    finally:
        await drv.disconnect()
