"""Unit tests for ``armdroid.telemetry`` — core provider and null backend.

Tests in this module do NOT require the ``armdroid[telemetry]`` extra
(opentelemetry-api/sdk).  They cover:

* :class:`~armdroid.telemetry.NullTelemetry` is the out-of-box default.
* :func:`~armdroid.telemetry.configure_telemetry` installs a custom
  provider and ``None`` resets it.
* :class:`~armdroid.telemetry.NullTelemetry` satisfies the
  :class:`~armdroid.telemetry.TelemetryProvider` Protocol.
* ``start_span`` is a valid context manager and does not raise.
* ``record_event`` is a callable no-op.
* Span name constants are non-empty strings.
"""

from __future__ import annotations

import pytest

from armdroid.telemetry import (
    SPAN_DRIVER_CONNECT,
    SPAN_DRIVER_DISCONNECT,
    SPAN_DRIVER_SEND,
    NullTelemetry,
    TelemetryProvider,
    configure_telemetry,
    get_telemetry,
)


@pytest.fixture(autouse=True)
def reset_global_provider() -> None:
    """Reset the module-level provider to NullTelemetry after every test."""
    yield  # type: ignore[misc]
    configure_telemetry(None)


# ---------------------------------------------------------------------------
# Default provider
# ---------------------------------------------------------------------------


def test_default_provider_is_null_telemetry() -> None:
    configure_telemetry(None)
    provider = get_telemetry()
    assert isinstance(provider, NullTelemetry)


def test_null_telemetry_satisfies_protocol() -> None:
    assert isinstance(NullTelemetry(), TelemetryProvider)


# ---------------------------------------------------------------------------
# configure_telemetry
# ---------------------------------------------------------------------------


def test_configure_replaces_default() -> None:
    class _StubTelemetry(NullTelemetry):
        pass

    stub = _StubTelemetry()
    configure_telemetry(stub)
    assert get_telemetry() is stub


def test_configure_none_resets_to_null() -> None:
    configure_telemetry(NullTelemetry())
    configure_telemetry(None)
    assert isinstance(get_telemetry(), NullTelemetry)


# ---------------------------------------------------------------------------
# NullTelemetry — start_span is a valid context manager
# ---------------------------------------------------------------------------


def test_null_start_span_is_context_manager() -> None:
    tel = NullTelemetry()
    ctx = tel.start_span(SPAN_DRIVER_CONNECT, port="COM3")
    # Must implement the context-manager protocol without raising.
    with ctx:
        pass  # No assertion needed — absence of exception is the contract.


def test_null_start_span_nested() -> None:
    tel = NullTelemetry()
    with tel.start_span(SPAN_DRIVER_CONNECT), tel.start_span(SPAN_DRIVER_SEND, dof=6):
        pass  # Nesting must not raise.


def test_null_start_span_exception_propagates() -> None:
    """Exceptions inside the span must not be swallowed."""
    tel = NullTelemetry()
    with pytest.raises(RuntimeError, match="test error"), tel.start_span(SPAN_DRIVER_CONNECT):
        raise RuntimeError("test error")


# ---------------------------------------------------------------------------
# NullTelemetry — record_event is a no-op
# ---------------------------------------------------------------------------


def test_null_record_event_does_not_raise() -> None:
    tel = NullTelemetry()
    tel.record_event("some.event", key="value", count=3)


# ---------------------------------------------------------------------------
# Span name constants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "constant",
    [SPAN_DRIVER_CONNECT, SPAN_DRIVER_DISCONNECT, SPAN_DRIVER_SEND],
)
def test_span_constant_is_non_empty_string(constant: str) -> None:
    assert isinstance(constant, str)
    assert constant.strip(), f"Span constant must not be empty or whitespace: {constant!r}"


def test_span_constants_are_distinct() -> None:
    constants = [SPAN_DRIVER_CONNECT, SPAN_DRIVER_DISCONNECT, SPAN_DRIVER_SEND]
    assert len(constants) == len(set(constants)), "Span constants must be unique"
