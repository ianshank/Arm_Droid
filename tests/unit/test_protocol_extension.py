"""Unit tests for the additive protocol-extension types.

Covers ``ArmState`` (frozen slots dataclass), ``ArmDriverError``, and
``ArmCommandRejected``. The full extended ``ArmDriverProtocol`` lifecycle
methods land in commit 5 alongside the new mock implementation.
"""

from __future__ import annotations

import pytest

from armdroid.protocols import ArmCommandRejected, ArmDriverError, ArmState


class TestArmState:
    """`ArmState` is a frozen, slotted dataclass with variable-length tuples."""

    def test_construct_holds_fields(self) -> None:
        state = ArmState(
            joint_positions=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            joint_velocities=(0.0,) * 7,
            is_moving=True,
            estop_active=False,
            timestamp_s=1234.5,
        )
        assert state.joint_positions == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        assert state.is_moving is True
        assert state.estop_active is False
        assert state.timestamp_s == pytest.approx(1234.5)

    def test_state_is_frozen(self) -> None:
        state = ArmState(
            joint_positions=(0.0,),
            joint_velocities=(0.0,),
            is_moving=False,
            estop_active=False,
            timestamp_s=0.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            state.is_moving = True  # type: ignore[misc]

    def test_state_uses_slots(self) -> None:
        state = ArmState(
            joint_positions=(0.0,),
            joint_velocities=(0.0,),
            is_moving=False,
            estop_active=False,
            timestamp_s=0.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            state.unknown_field = 42  # type: ignore[attr-defined]

    def test_supports_six_or_seven_joints(self) -> None:
        """Tuple shape is not pinned — both current 6-DoF and forthcoming
        7-DoF (with gripper) are valid."""
        six = ArmState((0.0,) * 6, (0.0,) * 6, False, False, 0.0)
        seven = ArmState((0.0,) * 7, (0.0,) * 7, False, False, 0.0)
        assert len(six.joint_positions) == 6
        assert len(seven.joint_positions) == 7


class TestArmDriverErrors:
    """Exception hierarchy: ArmCommandRejected is an ArmDriverError."""

    def test_command_rejected_is_driver_error(self) -> None:
        exc = ArmCommandRejected("joint[0] out of range")
        assert isinstance(exc, ArmDriverError)
        assert isinstance(exc, RuntimeError)

    def test_driver_error_is_runtime_error(self) -> None:
        exc = ArmDriverError("transport timeout")
        assert isinstance(exc, RuntimeError)

    def test_can_raise_and_catch_command_rejected_via_driver_error(
        self,
    ) -> None:
        with pytest.raises(ArmDriverError, match="bad shape"):
            raise ArmCommandRejected("bad shape")

    def test_can_raise_and_catch_command_rejected_directly(self) -> None:
        with pytest.raises(ArmCommandRejected, match="e-stop"):
            raise ArmCommandRejected("e-stop latched")
