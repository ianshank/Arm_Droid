"""Reusable in-memory TelemetryProvider stub for assertions.

Promoted from ``tests/unit/hardware/test_esp32_json_driver.py`` so every
suite that wants to assert on span emission shares one fake.

Usage::

    from armdroid.telemetry import configure_telemetry
    from tests.helpers.recording_telemetry import RecordingTelemetry

    tel = RecordingTelemetry()
    configure_telemetry(tel)
    try:
        # exercise code under test
        ...
    finally:
        configure_telemetry(None)

    assert "armdroid.driver.connect" in tel.spans
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from armdroid.telemetry import TelemetryProvider  # noqa: F401 - runtime_checkable Protocol


class RecordingTelemetry:
    """No-op TelemetryProvider that records ``start_span`` names + attrs.

    Conforms structurally to :class:`armdroid.telemetry.TelemetryProvider`
    via duck typing. We import the protocol above to make the conformance
    intent explicit and to surface upstream rename refactors.
    """

    def __init__(self) -> None:
        self.spans: list[str] = []
        self.span_attrs: list[dict[str, Any]] = []
        self.events: list[tuple[str, dict[str, Any]]] = []

    def start_span(self, name: str, **attrs: Any) -> Any:
        """Record the span name + attrs and return a no-op context manager."""
        self.spans.append(name)
        self.span_attrs.append(dict(attrs))

        @contextmanager
        def _ctx() -> Iterator[None]:
            yield

        return _ctx()

    def record_event(self, name: str, **attrs: Any) -> None:
        """Record an event for later assertion."""
        self.events.append((name, dict(attrs)))


__all__ = ["RecordingTelemetry"]
