"""Thin adapter exposing the v0.1.x arm-driver surface atop the modern protocol.

The modern ``ArmDriverProtocol`` uses explicit ``send_joint_positions(positions,
duration_s)`` and named lifecycle methods (``connect`` / ``disconnect``).  Code
written before the protocol extension may call the legacy interface instead:

* ``start()`` / ``stop()`` — lifecycle aliases for ``connect`` / ``disconnect``
* ``send_joint_command(angles)`` — motion without an explicit duration
* ``get_joint_states()`` — returns a raw numpy array instead of ``ArmState``
* ``open_gripper()`` / ``close_gripper()`` / ``home()`` — high-level primitives

:class:`LegacyArmDriverAdapter` wraps *any* ``ArmDriverProtocol``-conformant
object and delegates those calls by translating them to the modern surface.  A
:class:`DeprecationWarning` is raised on construction so consumers can migrate.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from armdroid.domain.protocols import ArmDriverProtocol
    from armdroid.domain.state import ArmState

# Default move duration injected by the legacy adapter when no duration is
# specified.  Kept deliberately slow so hardware does not receive an
# instantaneous jump command.
_LEGACY_MOVE_DURATION_S: float = 2.0


class LegacyArmDriverAdapter:
    """Wraps a modern :class:`ArmDriverProtocol` and exposes the v0.1.x surface.

    Args:
        driver: Any object that satisfies ``ArmDriverProtocol``.

    Raises:
        DeprecationWarning: Always, on construction.  Callers should migrate
            to the wrapped *driver* directly.

    Example::

        from armdroid.hardware.mock_arm_driver import MockArmDriver
        drv = MockArmDriver(cfg)
        legacy = LegacyArmDriverAdapter(drv)  # emits DeprecationWarning
        await legacy.start()
        await legacy.send_joint_command(np.zeros(6))
        await legacy.stop()
    """

    def __init__(self, driver: ArmDriverProtocol) -> None:
        warnings.warn(
            "LegacyArmDriverAdapter is deprecated and will be removed in v0.4.0. "
            "Use the wrapped driver directly via the ArmDriverProtocol interface.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._driver = driver

    # ---- Properties -------------------------------------------------------

    @property
    def dof(self) -> int:
        """Degrees of freedom of the wrapped driver."""
        return self._driver.dof

    @property
    def is_connected(self) -> bool:
        """Whether the transport is open."""
        return self._driver.is_connected

    # ---- Lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Open transport — legacy alias for :meth:`connect`."""
        await self._driver.connect()

    async def stop(self) -> None:
        """Close transport — legacy alias for :meth:`disconnect`."""
        await self._driver.disconnect()

    async def connect(self) -> None:
        """Open transport (modern interface — delegates to wrapped driver)."""
        await self._driver.connect()

    async def disconnect(self) -> None:
        """Close transport (modern interface — delegates to wrapped driver)."""
        await self._driver.disconnect()

    # ---- Motion -----------------------------------------------------------

    async def send_joint_command(self, target_angles: NDArray[np.float64]) -> None:
        """Command target joint angles using the legacy interface.

        Translates to :meth:`send_joint_positions` with the default move
        duration :data:`_LEGACY_MOVE_DURATION_S`.

        Args:
            target_angles: Target joint angles (radians), length == :attr:`dof`.
        """
        await self._driver.send_joint_positions(
            tuple(float(a) for a in target_angles),
            _LEGACY_MOVE_DURATION_S,
        )

    async def get_joint_states(self) -> NDArray[np.float64]:
        """Return current joint angles as a numpy array (legacy).

        Delegates to :meth:`read_state` and extracts
        :attr:`ArmState.joint_positions`.
        """
        state: ArmState = await self._driver.read_state()
        return np.asarray(state.joint_positions, dtype=np.float64)

    async def read_state(self) -> ArmState:
        """Return the latest arm telemetry (modern passthrough)."""
        return await self._driver.read_state()

    # ---- Gripper & home ---------------------------------------------------

    async def open_gripper(self) -> None:
        """Open the gripper fully — delegates to wrapped driver."""
        await self._driver.open_gripper()

    async def close_gripper(self) -> float:
        """Close the gripper — delegates to wrapped driver.

        Returns:
            Grip force in Newtons (implementation-defined).
        """
        return await self._driver.close_gripper()

    async def home(self) -> None:
        """Move arm to home position — delegates to wrapped driver."""
        await self._driver.home()

    # ---- Safety -----------------------------------------------------------

    async def emergency_stop(self) -> None:
        """Latch e-stop — delegates to wrapped driver."""
        await self._driver.emergency_stop()

    async def clear_emergency_stop(self) -> None:
        """Release e-stop — delegates to wrapped driver."""
        await self._driver.clear_emergency_stop()
