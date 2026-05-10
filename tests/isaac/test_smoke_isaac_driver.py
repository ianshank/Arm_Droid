"""Isaac Sim driver smoke test (PR-B B.16).

Runs locally only with ``ARMDROID_ISAAC_RUN=1 pytest tests/isaac``.
CI never installs the Isaac runtime because of the ~9 GB footprint and
the GPU requirement; this test exercises the full driver lifecycle on
a real CUDA box.

All tests share the session-scoped ``isaac_session_driver`` fixture
(see ``conftest.py``). Kit's AppLauncher is a process-wide singleton —
booting a fresh driver per test would crash on the second
``connect()`` (PR-11 review fix: copilot
``H-disconnect-doesnt-clear-flag`` /
``H-smoke-tests-singleton-fail``).

Asserts:
- ``connect()`` boots Kit + builds the articulation (verified by
  fixture setup completing successfully)
- ``read_state()`` returns ``ArmState`` with the configured DOF
- ``send_joint_positions`` accepts a valid command
- ``emergency_stop()`` latches; subsequent send raises ``ArmCommandRejected``
- ``clear_emergency_stop()`` releases
- ``disconnect()`` shuts down cleanly (verified by fixture teardown)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from armdroid.hardware.isaac_sim.driver import IsaacSimDriver


@pytest.mark.asyncio(loop_scope="session")
async def test_isaac_driver_is_connected(isaac_session_driver: IsaacSimDriver) -> None:
    """Session-scoped driver reports as connected after fixture setup."""
    assert isaac_session_driver.is_connected


@pytest.mark.asyncio(loop_scope="session")
async def test_isaac_driver_read_state(isaac_session_driver: IsaacSimDriver) -> None:
    """read_state returns the configured-DOF state vector."""
    from armdroid.domain.state import ArmState

    state = await isaac_session_driver.read_state()
    assert isinstance(state, ArmState)
    assert len(state.joint_positions) == isaac_session_driver.dof
    assert len(state.joint_velocities) == isaac_session_driver.dof
    assert state.estop_active is False


@pytest.mark.asyncio(loop_scope="session")
async def test_isaac_driver_estop_blocks_motion(
    isaac_session_driver: IsaacSimDriver,
) -> None:
    """emergency_stop() latches; subsequent send is rejected.

    Restores e-stop state at the end so subsequent tests in this
    session see the driver in a clean, non-latched state.
    """
    from armdroid.domain.errors import ArmCommandRejected

    drv = isaac_session_driver
    try:
        await drv.emergency_stop()
        with pytest.raises(ArmCommandRejected, match="emergency"):
            await drv.send_joint_positions((0.0,) * drv.dof, duration_s=1.0)
        await drv.clear_emergency_stop()
        # After clear, motion should succeed again.
        await drv.send_joint_positions((0.0,) * drv.dof, duration_s=1.0)
    finally:
        # Defensive: ensure latch is released even if assertions failed.
        await drv.clear_emergency_stop()
