"""Telemetry seam for armdroid — no-op default, optional OTel provider.

This package provides a minimal instrumentation contract
(:class:`TelemetryProvider`) that all driver, planner, and environment
implementations can consume without taking a hard dependency on any
specific observability backend.

Default behaviour
-----------------
:func:`get_telemetry` returns a :class:`NullTelemetry` instance.  All
``start_span`` calls return :func:`contextlib.nullcontext` and all
``record_event`` calls are no-ops.  The import footprint is zero — just
the standard library.

Using the OTel backend
----------------------
Install the ``armdroid[telemetry]`` extra and wire up the provider at
application startup::

    from armdroid.telemetry import configure_telemetry
    from armdroid.telemetry.otel import OtelTelemetry

    configure_telemetry(OtelTelemetry())

Adding spans to a new subsystem
--------------------------------
Import the pre-defined span name constants so names are never hardcoded
at call sites::

    from armdroid.telemetry import get_telemetry, SPAN_DRIVER_CONNECT

    with get_telemetry().start_span(SPAN_DRIVER_CONNECT, port=port_path):
        ...  # operation to instrument
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Span name constants
# ---------------------------------------------------------------------------

#: Span that covers the full ``Esp32JsonDriver.connect()`` call.
SPAN_DRIVER_CONNECT: str = "armdroid.driver.connect"

#: Span that covers the full ``Esp32JsonDriver.disconnect()`` call.
SPAN_DRIVER_DISCONNECT: str = "armdroid.driver.disconnect"

#: Span that covers ``Esp32JsonDriver.send_joint_positions()`` including
#: local validation and the firmware ack round-trip.
SPAN_DRIVER_SEND: str = "armdroid.driver.send_joint_positions"

# Environment span constants — emitted by every ArmEnvironmentBase subclass
# (Tower of Hanoi, Laundry Sorting today; PR-B's SoArmReachIsaacEnv next).

#: Span covering :meth:`ArmEnvironmentBase.reset`.
SPAN_ENV_RESET: str = "armdroid.env.reset"

#: Span covering :meth:`ArmEnvironmentBase.step`.
SPAN_ENV_STEP: str = "armdroid.env.step"

#: Span covering :meth:`ArmEnvironmentBase.render`.
SPAN_ENV_RENDER: str = "armdroid.env.render"

#: Span covering :meth:`ArmEnvironmentBase.close`.
SPAN_ENV_CLOSE: str = "armdroid.env.close"

# Agent span constants — emitted by every ArmRLAgentProtocol implementation
# (SACAgent today; PR-B's RslRlPpoAgent next).

#: Span covering :meth:`ArmRLAgentProtocol.build` (model construction).
SPAN_AGENT_BUILD: str = "armdroid.agent.build"

#: Span covering :meth:`ArmRLAgentProtocol.train` (full training run).
SPAN_AGENT_TRAIN: str = "armdroid.agent.train"

#: Span covering :meth:`ArmRLAgentProtocol.predict` (one inference call).
SPAN_AGENT_PREDICT: str = "armdroid.agent.predict"

#: Span covering :meth:`ArmRLAgentProtocol.save`.
SPAN_AGENT_SAVE: str = "armdroid.agent.save"

#: Span covering :meth:`ArmRLAgentProtocol.load`.
SPAN_AGENT_LOAD: str = "armdroid.agent.load"

# Vec-path span constants (F1) — emitted by VecArmEnvironmentProtocol
# implementations and VecArmRLAgentProtocol agents.

#: Span covering :meth:`VecArmRLAgentProtocol.build_vec`.
SPAN_AGENT_BUILD_VEC: str = "armdroid.agent.build_vec"

#: Span covering :meth:`VecArmRLAgentProtocol.train_vec`.
SPAN_AGENT_TRAIN_VEC: str = "armdroid.agent.train_vec"

#: Span covering :meth:`VecArmEnvironmentProtocol.reset`.
SPAN_ENV_VEC_RESET: str = "armdroid.env.vec_reset"

#: Span covering :meth:`VecArmEnvironmentProtocol.step`.
SPAN_ENV_VEC_STEP: str = "armdroid.env.vec_step"

#: Span covering :meth:`VecArmEnvironmentProtocol.close`.
SPAN_ENV_VEC_CLOSE: str = "armdroid.env.vec_close"

#: Span covering the AppLauncher coordination + ``ManagerBasedRLEnv`` boot
#: inside :meth:`SoArmReachIsaacVecEnv._ensure_built`.
SPAN_ENV_VEC_KIT_BOOT: str = "armdroid.env.vec_kit_boot"


# ---------------------------------------------------------------------------
# TelemetryProvider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TelemetryProvider(Protocol):
    """Minimal telemetry contract for armdroid subsystems.

    Implementors must provide:

    * :meth:`start_span` — context manager that delimits a named
      operation.  The context manager is entered immediately; attributes
      are attached to the span at creation time.
    * :meth:`record_event` — record a point-in-time event within the
      current span (or globally if no span is active).
    """

    def start_span(
        self,
        name: str,
        **attrs: Any,
    ) -> AbstractContextManager[None]:
        """Return a context manager that delimits a named span.

        Args:
            name: Span name.  Use the pre-defined ``SPAN_*`` constants
                so names are consistent across implementations.
            **attrs: Arbitrary key-value attributes attached to the span.

        Returns:
            An ``AbstractContextManager[None]`` whose ``__enter__`` marks
            span start and ``__exit__`` marks span end.
        """
        ...

    def record_event(self, name: str, **attrs: Any) -> None:
        """Record a named point-in-time event.

        Args:
            name: Event name.
            **attrs: Arbitrary key-value attributes.
        """
        ...


# ---------------------------------------------------------------------------
# NullTelemetry — zero-overhead default
# ---------------------------------------------------------------------------


class NullTelemetry:
    """No-op telemetry provider.  Zero overhead; does nothing."""

    def start_span(
        self,
        name: str,
        **attrs: Any,
    ) -> AbstractContextManager[None]:
        """Return :func:`contextlib.nullcontext` — zero overhead."""
        return nullcontext()

    def record_event(self, name: str, **attrs: Any) -> None:
        """No-op."""


# ---------------------------------------------------------------------------
# Module-level provider registry
# ---------------------------------------------------------------------------

_provider: TelemetryProvider = NullTelemetry()


def configure_telemetry(provider: TelemetryProvider | None) -> None:
    """Set the module-level telemetry provider.

    Call this once at application startup (e.g. in ``armdroid.main``)
    after constructing the desired backend.  Passing ``None`` resets the
    provider back to :class:`NullTelemetry`.

    Args:
        provider: The backend to install, or ``None`` to restore the
            no-op default.

    Example::

        from armdroid.telemetry import configure_telemetry
        from armdroid.telemetry.otel import OtelTelemetry

        configure_telemetry(OtelTelemetry())
    """
    global _provider
    _provider = provider if provider is not None else NullTelemetry()


def get_telemetry() -> TelemetryProvider:
    """Return the currently configured telemetry provider.

    Callers should not cache the return value across calls — the provider
    may be replaced at runtime during tests or reconfiguration.

    Returns:
        The active :class:`TelemetryProvider` (default: :class:`NullTelemetry`).
    """
    return _provider


__all__ = [
    "SPAN_AGENT_BUILD",
    "SPAN_AGENT_BUILD_VEC",
    "SPAN_AGENT_LOAD",
    "SPAN_AGENT_PREDICT",
    "SPAN_AGENT_SAVE",
    "SPAN_AGENT_TRAIN",
    "SPAN_AGENT_TRAIN_VEC",
    "SPAN_DRIVER_CONNECT",
    "SPAN_DRIVER_DISCONNECT",
    "SPAN_DRIVER_SEND",
    "SPAN_ENV_CLOSE",
    "SPAN_ENV_RENDER",
    "SPAN_ENV_RESET",
    "SPAN_ENV_STEP",
    "SPAN_ENV_VEC_CLOSE",
    "SPAN_ENV_VEC_KIT_BOOT",
    "SPAN_ENV_VEC_RESET",
    "SPAN_ENV_VEC_STEP",
    "NullTelemetry",
    "TelemetryProvider",
    "configure_telemetry",
    "get_telemetry",
]
