"""Tests for the expanded :mod:`armdroid.domain.errors` hierarchy."""

from __future__ import annotations

import pytest

from armdroid import (
    ArmCommandRejected,
    ArmDriverError,
    ArmDroidError,
    ConfigError,
    PerceptionError,
    PlanningError,
)


@pytest.mark.parametrize(
    "exc_cls",
    [
        ArmCommandRejected,
        ArmDriverError,
        ConfigError,
        PerceptionError,
        PlanningError,
    ],
)
def test_all_errors_descend_from_armdroid_root(exc_cls: type[Exception]) -> None:
    """Every armdroid exception subclasses :class:`ArmDroidError`."""
    assert issubclass(exc_cls, ArmDroidError)


@pytest.mark.parametrize(
    "exc_cls",
    [ArmDroidError, ArmDriverError, ArmCommandRejected, ConfigError],
)
def test_runtime_error_compat_preserved(exc_cls: type[Exception]) -> None:
    """Pre-v0.2 ``except RuntimeError`` callers still match new errors."""
    assert issubclass(exc_cls, RuntimeError)


def test_command_rejected_still_descends_from_driver_error() -> None:
    """ArmCommandRejected MUST remain a subclass of ArmDriverError.

    Existing code (and tests) catch ``ArmDriverError`` to handle both
    transport failures and rejected commands.
    """
    assert issubclass(ArmCommandRejected, ArmDriverError)


def test_subsystem_errors_are_distinct() -> None:
    """ConfigError, PerceptionError, PlanningError are siblings, not nested."""
    assert not issubclass(ConfigError, ArmDriverError)
    assert not issubclass(PerceptionError, ArmDriverError)
    assert not issubclass(PlanningError, ArmDriverError)
    assert not issubclass(ConfigError, PerceptionError)


def test_can_raise_and_catch_at_root() -> None:
    """A single ``except ArmDroidError`` handler matches every subclass."""
    for exc in [
        ConfigError("bad config"),
        ArmDriverError("transport"),
        ArmCommandRejected("rejected"),
        PerceptionError("camera"),
        PlanningError("planner"),
    ]:
        with pytest.raises(ArmDroidError):
            raise exc
