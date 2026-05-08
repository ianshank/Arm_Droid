"""Unit tests for :class:`~armdroid.telemetry.otel.OtelTelemetry`.

All tests are skipped automatically when ``opentelemetry-sdk`` is not
installed (i.e. the ``armdroid[telemetry]`` extra is absent).

When the extra *is* installed the tests verify:

* :class:`~armdroid.telemetry.otel.OtelTelemetry` satisfies the
  :class:`~armdroid.telemetry.TelemetryProvider` protocol.
* ``start_span`` returns a valid context manager and does not raise.
* ``record_event`` does not raise.
* Driver spans (connect, send, disconnect) are exported to an
  in-memory exporter and appear in the recorded output.
"""

from __future__ import annotations

import importlib.util

import pytest

# ---------------------------------------------------------------------------
# Module-level skip guard
# ---------------------------------------------------------------------------

try:
    # find_spec with a dotted name imports the parent package first;
    # if 'opentelemetry' is absent this raises ModuleNotFoundError.
    _otel_available: bool = importlib.util.find_spec("opentelemetry.sdk") is not None
except ModuleNotFoundError:
    _otel_available = False

pytestmark = pytest.mark.skipif(
    not _otel_available,
    reason="opentelemetry-sdk not installed (pip install -e '.[telemetry]')",
)


# ---------------------------------------------------------------------------
# Lazy imports (only evaluated when OTel is available)
# ---------------------------------------------------------------------------

if _otel_available:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from armdroid.telemetry import (
        TelemetryProvider,
        configure_telemetry,
    )
    from armdroid.telemetry.otel import OtelTelemetry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_global_provider() -> None:
    """Reset armdroid global provider after every test."""
    yield  # type: ignore[misc]
    from armdroid.telemetry import configure_telemetry as _reset

    _reset(None)


@pytest.fixture
def in_memory_otel() -> tuple[OtelTelemetry, InMemorySpanExporter]:
    """Set up an OTel TracerProvider with an in-memory exporter.

    Yields an ``(OtelTelemetry, InMemorySpanExporter)`` pair.

    The ``TracerProvider`` is passed *directly* to :class:`OtelTelemetry`
    rather than installed globally.  This avoids the OTel SDK restriction
    "Overriding of current TracerProvider is not allowed" and keeps each
    test fully isolated.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        yield OtelTelemetry(tracer_provider=provider), exporter
    finally:
        provider.shutdown()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_otel_telemetry_satisfies_protocol() -> None:
    assert isinstance(OtelTelemetry(), TelemetryProvider)


# ---------------------------------------------------------------------------
# start_span
# ---------------------------------------------------------------------------


def test_start_span_is_context_manager(
    in_memory_otel: tuple[OtelTelemetry, InMemorySpanExporter],
) -> None:
    tel, _ = in_memory_otel
    ctx = tel.start_span("test.span", key="value")
    with ctx:
        pass  # Must not raise.


def test_start_span_exception_propagates(
    in_memory_otel: tuple[OtelTelemetry, InMemorySpanExporter],
) -> None:
    tel, _ = in_memory_otel
    with pytest.raises(ValueError, match="boom"), tel.start_span("test.span"):
        raise ValueError("boom")


# ---------------------------------------------------------------------------
# record_event
# ---------------------------------------------------------------------------


def test_record_event_does_not_raise(
    in_memory_otel: tuple[OtelTelemetry, InMemorySpanExporter],
) -> None:
    tel, _ = in_memory_otel
    with tel.start_span("test.outer"):
        tel.record_event("some.event", count=1)


def test_record_event_outside_span_does_not_raise(
    in_memory_otel: tuple[OtelTelemetry, InMemorySpanExporter],
) -> None:
    tel, _ = in_memory_otel
    # No active span — should silently no-op (OTel absorbs add_event on
    # INVALID_SPAN without raising).
    tel.record_event("orphan.event")


# ---------------------------------------------------------------------------
# In-memory span capture — driver span names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "span_name",
    [
        "armdroid.driver.connect",
        "armdroid.driver.send_joint_positions",
        "armdroid.driver.disconnect",
    ],
)
def test_driver_span_recorded_in_exporter(
    in_memory_otel: tuple[OtelTelemetry, InMemorySpanExporter],
    span_name: str,
) -> None:
    tel, exporter = in_memory_otel
    configure_telemetry(tel)
    with tel.start_span(span_name):
        pass
    finished = exporter.get_finished_spans()
    span_names = [s.name for s in finished]
    assert span_name in span_names, (
        f"Expected span '{span_name}' to appear in exported spans; got: {span_names}"
    )


# ---------------------------------------------------------------------------
# ImportError when OTel absent (tested by module fallback logic)
# ---------------------------------------------------------------------------


def test_otel_telemetry_available_when_sdk_installed() -> None:
    """Smoke-test: constructing OtelTelemetry must succeed when OTel is present."""
    tel = OtelTelemetry()
    assert tel is not None
