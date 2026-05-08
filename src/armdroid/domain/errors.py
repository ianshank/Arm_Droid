"""Exception hierarchy for the armdroid platform.

Currently scoped to driver-layer faults; Phase 4 will introduce a unified
``ArmDroidError`` root plus ``ConfigError``, ``PerceptionError``, and
formalise ``PlanningError`` here. For v0.2.0 this module hosts the
hardware-driver exceptions surfaced by the public API.
"""

from __future__ import annotations


class ArmDriverError(RuntimeError):
    """Base exception for arm driver failures (transport, timeout, protocol).

    Raised when the driver cannot complete an operation due to a transport
    or firmware fault — the host code should consider the arm in an unknown
    state and either reconnect or abort the current task.
    """


class ArmCommandRejected(ArmDriverError):  # noqa: N818
    """Raised when a command is rejected, either locally or by the firmware.

    Raised before transmission for locally detectable violations (joint-limit
    breach, NaN or wrong-length command vector, non-positive duration, or
    exceeding the wire line-length cap) and after transmission when the
    firmware returns a NAK (e.g. ``out_of_range`` or ``estop_latched``).
    Distinct from :class:`ArmDriverError` because the arm is *not* in a
    faulted state — the caller can simply correct the command and retry.
    """


__all__ = ["ArmCommandRejected", "ArmDriverError"]
