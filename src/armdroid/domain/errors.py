"""Exception hierarchy for the armdroid platform.

All armdroid-raised exceptions inherit from :class:`ArmDroidError` so that
callers can install a single ``except ArmDroidError`` handler at the
process boundary. Subsystem-specific subclasses let callers respond to
particular failure modes (driver vs config vs perception vs planning).

Hierarchy::

    ArmDroidError (RuntimeError)
    ├── ConfigError       — invalid configuration / YAML overlay merge failure
    ├── ArmDriverError    — transport / firmware fault (arm in unknown state)
    │   └── ArmCommandRejected — locally-detected or NAK'd command (recoverable)
    ├── PerceptionError   — depth / detection / pose pipeline failure
    └── PlanningError     — symbolic planner / replanner failure

``ArmDriverError`` and ``ArmCommandRejected`` retain their pre-v0.2.0
positional behaviour: both still inherit (transitively) from
:class:`RuntimeError`, so ``except RuntimeError`` handlers in legacy
calling code continue to match.
"""

from __future__ import annotations


class ArmDroidError(RuntimeError):
    """Root of the armdroid exception hierarchy.

    Inherits from :class:`RuntimeError` so that pre-v0.2.0 callers using
    ``except RuntimeError`` continue to function. New code should catch
    :class:`ArmDroidError` (or a subclass) for armdroid-specific failures.
    """


class ConfigError(ArmDroidError):
    """Raised for configuration validation or overlay-merge failures.

    Wraps Pydantic ``ValidationError`` and YAML parse errors at the
    :mod:`armdroid.config.loader` boundary so the public surface does not
    leak third-party exception types.
    """


class ArmDriverError(ArmDroidError):
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


class PerceptionError(ArmDroidError):
    """Raised for failures in the perception pipeline.

    Surfaced by the depth processor, object detector, pose estimator, or
    state extractor when input data is malformed or a backend fails to
    initialise (e.g. RealSense pipeline cannot start).
    """


class PlanningError(ArmDroidError):
    """Raised for symbolic-planner or LLM-replanner failures.

    Wraps pyperplan failures and replanner-backend errors at the
    :mod:`armdroid.planning` boundary.
    """


__all__ = [
    "ArmCommandRejected",
    "ArmDriverError",
    "ArmDroidError",
    "ConfigError",
    "PerceptionError",
    "PlanningError",
]
