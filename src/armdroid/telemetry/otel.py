"""OpenTelemetry-backed telemetry provider for armdroid.

This module implements :class:`~armdroid.telemetry.TelemetryProvider`
using ``opentelemetry-api``.  Install the ``armdroid[telemetry]`` extra
to make the provider importable at runtime::

    pip install -e ".[telemetry]"

Without the extra this module is still *importable* — constructing
:class:`OtelTelemetry` raises :exc:`ImportError` at that point, so the
import itself never fails.  This keeps startup code safe even in
environments where OTel is absent.

Wiring example::

    from armdroid.telemetry import configure_telemetry
    from armdroid.telemetry.otel import OtelTelemetry

    configure_telemetry(OtelTelemetry())
"""

from __future__ import annotations

import importlib.util
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Import OTel types for type annotations only — never executed at runtime
    # unless opentelemetry-api is installed.
    from opentelemetry.trace import Tracer
    from opentelemetry.trace import TracerProvider as OtelTracerProvider

# Check availability without binding a module-level variable.  Using find_spec
# avoids version-specific ``type: ignore`` or Any gymnastics: mypy 2.0 strict
# flags both ``type: ignore[assignment]`` (unused-ignore) and a try/except
# rebind of an Any-annotated module variable (no-redef).
_OTEL_AVAILABLE: bool = importlib.util.find_spec("opentelemetry") is not None


# ---------------------------------------------------------------------------
# Internal context-manager adapter
# ---------------------------------------------------------------------------


class _OtelSpanContext:
    """Thin adapter that exposes an OTel span as ``AbstractContextManager[None]``.

    OTel's ``Tracer.start_as_current_span`` returns a context manager
    that yields a :class:`opentelemetry.trace.Span`, but our
    :class:`~armdroid.telemetry.TelemetryProvider` protocol promises
    ``AbstractContextManager[None]`` (callers never need the span
    object).  This adapter drops the yielded value.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __enter__(self) -> None:
        self._inner.__enter__()
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool | None:
        result: Any = self._inner.__exit__(exc_type, exc_val, exc_tb)
        # __exit__ returning a truthy value suppresses the exception.
        return bool(result) if result else None


# ---------------------------------------------------------------------------
# OtelTelemetry
# ---------------------------------------------------------------------------


class OtelTelemetry:
    """OpenTelemetry-backed :class:`~armdroid.telemetry.TelemetryProvider`.

    Delegates to a named OTel :class:`~opentelemetry.trace.Tracer` for
    span creation and to the current span for event recording.

    Args:
        tracer_name: Tracer name passed to ``get_tracer``.
            Defaults to ``"armdroid"``.
        tracer_provider: Optional OTel ``TracerProvider`` to use instead of
            the global one.  Passing an explicit provider makes the instance
            fully self-contained and is required in unit tests to avoid
            the "Overriding of current TracerProvider is not allowed"
            restriction imposed by the OTel SDK.

    Raises:
        ImportError: If ``opentelemetry-api`` is not installed.
    """

    def __init__(
        self,
        tracer_name: str = "armdroid",
        tracer_provider: OtelTracerProvider | None = None,
    ) -> None:
        if not _OTEL_AVAILABLE:
            msg = (
                "opentelemetry-api is not installed. "
                "Install armdroid with the [telemetry] extra: "
                "pip install -e '.[telemetry]'"
            )
            raise ImportError(msg)
        import opentelemetry.trace as _otel_trace_local  # succeeds: _OTEL_AVAILABLE guard above
        if tracer_provider is not None:
            self._tracer: Tracer = tracer_provider.get_tracer(tracer_name)
        else:
            self._tracer = _otel_trace_local.get_tracer(tracer_name)

    def start_span(
        self,
        name: str,
        **attrs: Any,
    ) -> AbstractContextManager[None]:
        """Start a new OTel span as a context manager.

        Args:
            name: Span name.
            **attrs: Span attributes forwarded to OTel.

        Returns:
            An :class:`AbstractContextManager[None]` wrapping the OTel span.
        """
        return _OtelSpanContext(self._tracer.start_as_current_span(name, attributes=attrs))

    def record_event(self, name: str, **attrs: Any) -> None:
        """Add an event to the currently active OTel span.

        If no span is active the event is silently discarded (OTel
        ``INVALID_SPAN`` absorbs ``add_event`` without raising).

        Args:
            name: Event name.
            **attrs: Event attributes forwarded to OTel.
        """
        if _OTEL_AVAILABLE:
            import opentelemetry.trace as _otel_trace_local  # succeeds when _OTEL_AVAILABLE is True
            _otel_trace_local.get_current_span().add_event(name, attributes=attrs)


__all__ = ["OtelTelemetry"]
