"""Hardware validation tests for multiple ESP32 transports (USB, TCP, BLE).

Tests connection, HMAC auth, home pose, and emergency-stop mechanisms
over the supported physical transports.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator

import pytest

from armdroid.config.schema import ArmSettings
from armdroid.domain.errors import ArmDriverError
from armdroid.hardware.esp32 import Esp32JsonDriver


# Parameterize over the 3 transports.
# Note: For TCP and BLE, the device must actually be configured and reachable
# on the network/Bluetooth adapter. Tests will attempt connection and fail gracefully
# if the transport isn't physically available.
@pytest.fixture(params=["serial", "tcp", "ble"])
def transport_settings(request: pytest.FixtureRequest, hil_settings: ArmSettings) -> ArmSettings:
    transport_protocol = request.param
    base = hil_settings.model_copy()
    patched_transport = base.arm.transport.model_copy(update={"protocol": transport_protocol})
    patched_arm = base.arm.model_copy(update={"transport": patched_transport})
    return base.model_copy(update={"arm": patched_arm})

@pytest.fixture
async def transport_driver(
    transport_settings: ArmSettings,
) -> AsyncGenerator[Esp32JsonDriver, None]:
    drv = Esp32JsonDriver(transport_settings.arm)
    try:
        # Give transports like BLE a little longer to discover/connect
        await asyncio.wait_for(drv.connect(), timeout=10.0)
    except Exception as e:
        protocol = transport_settings.arm.transport.protocol
        pytest.skip(f"Transport {protocol} unavailable or failed to connect: {e}")
        return

    try:
        yield drv
    finally:
        with contextlib.suppress(Exception):
            await drv.clear_emergency_stop()
        await drv.disconnect()


@pytest.mark.hardware
@pytest.mark.asyncio
async def test_hmac_auth_and_connect(transport_driver: Esp32JsonDriver) -> None:
    """Validate that the connection establishes securely and state is received."""
    assert transport_driver.is_connected
    state = await transport_driver.read_state()
    assert state is not None
    # We should have valid joint positions for 6-DoF
    assert len(state.joint_positions) == transport_driver.dof


@pytest.mark.hardware
@pytest.mark.asyncio
async def test_emergency_stop_latching(transport_driver: Esp32JsonDriver) -> None:
    """Verify that E-Stop latches immediately and rejects further commands."""
    assert transport_driver.is_connected

    # Issue E-stop
    await transport_driver.emergency_stop()

    # State should reflect E-stop active
    # We poll slightly as the state frame might take a tick to arrive
    for _ in range(20):
        state = await transport_driver.read_state()
        if state.estop_active:
            break
        await asyncio.sleep(0.05)

    assert state.estop_active, "E-stop did not latch on firmware."

    # Commands should be rejected while latched
    home_pos = tuple(0.0 for _ in range(transport_driver.dof))
    with pytest.raises(ArmDriverError) as exc:
        await transport_driver.send_joint_positions(home_pos, duration_s=1.0)

    assert "estop" in str(exc.value).lower()

    # Unlatch
    await transport_driver.clear_emergency_stop()
    for _ in range(20):
        state = await transport_driver.read_state()
        if not state.estop_active:
            break
        await asyncio.sleep(0.05)

    assert not state.estop_active, "Failed to clear E-stop."


@pytest.mark.hardware
@pytest.mark.asyncio
async def test_home_pose_command(transport_driver: Esp32JsonDriver) -> None:
    """Verify that commanding the home pose returns the arm to a stable state."""
    # Move to home pose over 1.0 second
    home_pos = tuple(0.0 for _ in range(transport_driver.dof))

    await transport_driver.send_joint_positions(home_pos, duration_s=1.0)

    # Wait for movement to complete
    await asyncio.sleep(1.2)

    state = await transport_driver.read_state()
    assert not state.is_moving

    # Check that positions are roughly near home (allow some tolerance for PID steady state)
    for pos in state.joint_positions:
        assert abs(pos) < 0.1, f"Joint did not reach home pose, stuck at {pos}"
