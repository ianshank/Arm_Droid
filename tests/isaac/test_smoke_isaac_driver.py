"""Isaac Sim driver smoke test (PR-B B.16).

Runs locally only with ``ARMDROID_ISAAC_RUN=1 pytest tests/isaac``.
CI never installs the Isaac runtime because of the ~9 GB footprint and
the GPU requirement; this test exercises the full driver lifecycle on
a real CUDA box.

Asserts:
- ``connect()`` boots Kit + builds the articulation
- ``read_state()`` returns ``ArmState`` with the configured DOF
- ``send_joint_positions`` accepts a valid command
- ``emergency_stop()`` latches; subsequent send raises ``ArmCommandRejected``
- ``clear_emergency_stop()`` releases
- ``disconnect()`` shuts down cleanly
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass


@pytest.mark.asyncio
async def test_isaac_driver_connect_disconnect(isaac_available: None) -> None:
    """Connect → disconnect cycle without errors."""
    from armdroid.config.schema import ArmSettings
    from armdroid.hardware.isaac_sim import IsaacSimDriver

    cfg = ArmSettings(arm_driver_kind="isaac_sim")
    drv = IsaacSimDriver(cfg.arm)
    await drv.connect()
    assert drv.is_connected
    await drv.disconnect()
    assert not drv.is_connected


@pytest.mark.asyncio
async def test_isaac_driver_read_state(isaac_available: None) -> None:
    """read_state returns the configured-DOF state vector."""
    from armdroid.config.schema import ArmSettings
    from armdroid.domain.state import ArmState
    from armdroid.hardware.isaac_sim import IsaacSimDriver

    cfg = ArmSettings(arm_driver_kind="isaac_sim")
    drv = IsaacSimDriver(cfg.arm)
    await drv.connect()
    try:
        state = await drv.read_state()
        assert isinstance(state, ArmState)
        assert len(state.joint_positions) == cfg.arm.dof
        assert len(state.joint_velocities) == cfg.arm.dof
        assert state.estop_active is False
    finally:
        await drv.disconnect()


@pytest.mark.asyncio
async def test_isaac_driver_estop_blocks_motion(isaac_available: None) -> None:
    """emergency_stop() latches; subsequent send is rejected."""
    from armdroid.config.schema import ArmSettings
    from armdroid.domain.errors import ArmCommandRejected
    from armdroid.hardware.isaac_sim import IsaacSimDriver

    cfg = ArmSettings(arm_driver_kind="isaac_sim")
    drv = IsaacSimDriver(cfg.arm)
    await drv.connect()
    try:
        await drv.emergency_stop()
        with pytest.raises(ArmCommandRejected, match="emergency"):
            await drv.send_joint_positions((0.0,) * cfg.arm.dof, duration_s=1.0)
        await drv.clear_emergency_stop()
        # After clear, motion should succeed again.
        await drv.send_joint_positions((0.0,) * cfg.arm.dof, duration_s=1.0)
    finally:
        await drv.disconnect()
