"""HIL protocol-edge tests for the ESP32 JSON driver.

Exercises firmware error paths that cannot be reached through the public
``ArmDriverProtocol`` surface (because host-side validation pre-empts
them) by calling driver internals directly.  Also tests oversized-line
overflow handling and the ``evt:fault`` dispatch code path.

Run with::

    ARMDROID_HIL_RUN=1 pytest tests/hardware/test_real_esp32_protocol_edges.py \\
        -m hardware -v

All tests require a real ESP32 with armdroid firmware attached and
``ARMDROID_HIL_RUN=1`` set.  Fault-injection tests additionally require
``ARMDROID_HIL_FAULT_INJECT=1``.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from armdroid.config.schema import ArmSettings
from armdroid.domain.errors import ArmCommandRejected, ArmDriverError
from armdroid.hardware.esp32 import Esp32JsonDriver

# ---------------------------------------------------------------------------
# nak: out_of_range
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nak_out_of_range_propagated(
    hil_driver: Esp32JsonDriver,
    hil_settings: ArmSettings,
) -> None:
    """Firmware rejects an out-of-range joint value; driver surfaces it.

    Bypasses host-side validation via ``_send_and_await_ack`` so the
    command reaches the firmware with a value that violates joint limits.
    The firmware replies with ``nak:out_of_range`` which the driver maps
    to :class:`~armdroid.domain.errors.ArmCommandRejected`.
    """
    limits = hil_settings.arm.joint_limits
    # First joint pushed 1.0 rad beyond its max — guaranteed out of range.
    q_bad = [limits[0].max_rad + 1.0] + [0.0] * (hil_settings.arm.dof - 1)
    with pytest.raises((ArmCommandRejected, ArmDriverError)):
        await hil_driver._send_and_await_ack(  # type: ignore[attr-defined]
            cmd="set_joints",
            payload={"q": q_bad, "dur_ms": 500},
        )


# ---------------------------------------------------------------------------
# nak: bad_joint_count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nak_bad_joint_count_propagated(
    hil_driver: Esp32JsonDriver,
    hil_settings: ArmSettings,
) -> None:
    """Firmware rejects a ``q`` vector with the wrong joint count.

    Sends ``dof + 1`` zeros — the firmware's shape-check fires before
    it tries to move, so no physical motion occurs.  Driver maps the
    ``nak:bad_joint_count`` reply to
    :class:`~armdroid.domain.errors.ArmCommandRejected`.
    """
    dof = hil_settings.arm.dof
    q_wrong = [0.0] * (dof + 1)
    with pytest.raises((ArmCommandRejected, ArmDriverError)):
        await hil_driver._send_and_await_ack(  # type: ignore[attr-defined]
            cmd="set_joints",
            payload={"q": q_wrong, "dur_ms": 500},
        )


# ---------------------------------------------------------------------------
# max_line_bytes overflow — firmware drops oversized line silently
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oversized_line_dropped_silently(
    hil_driver: Esp32JsonDriver,
    hil_settings: ArmSettings,
) -> None:
    """Firmware discards a line exceeding ``max_line_bytes`` without crashing.

    Writes a line that is 2x the configured limit directly to the
    underlying ``serial.Serial`` port object, bypassing all driver
    framing.  After the overflow the driver must still exchange a valid
    command, confirming the firmware's parser recovered cleanly.
    """
    max_bytes = hil_settings.arm.transport.max_line_bytes
    # Build a deliberately oversized line (2x limit + newline delimiter).
    oversized = b"x" * (max_bytes * 2) + b"\n"

    port = hil_driver._port  # type: ignore[attr-defined]
    assert port is not None, "hil_driver must be connected before the test runs"
    await asyncio.to_thread(port.write, oversized)
    await asyncio.to_thread(port.flush)

    # Allow the firmware a moment to discard the bad line.
    await asyncio.sleep(0.15)

    # A valid ping must still be ack'd — firmware parser is healthy.
    await hil_driver._send_and_await_ack(  # type: ignore[attr-defined]
        cmd="ping",
        payload={},
    )


# ---------------------------------------------------------------------------
# evt:fault dispatch — exercises _dispatch_line without needing firmware fault
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evt_fault_dispatch_does_not_crash(
    hil_driver: Esp32JsonDriver,
) -> None:
    """Driver handles a synthetic ``evt:fault`` frame without crashing.

    Directly invokes ``_dispatch_line`` with a crafted fault-event frame
    to exercise the ``evt`` branch of the dispatch logic.  This does NOT
    write anything to the firmware; it tests the host-side parser only.
    After processing the synthetic frame the real connection must remain
    fully operational (verified by a live ping).
    """
    fault_frame = (
        json.dumps(
            {
                "t": "evt",
                "kind": "fault",
                "code": "test_fault",
                "msg": "synthetic — injected by HIL test suite",
                "ts": 0.0,
            },
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    # Inject into the dispatch path directly — no UART write needed.
    hil_driver._dispatch_line(fault_frame)  # type: ignore[attr-defined]

    # Driver and connection must survive the unexpected event.
    await hil_driver._send_and_await_ack(  # type: ignore[attr-defined]
        cmd="ping",
        payload={},
    )


@pytest.mark.asyncio
async def test_evt_boot_dispatch_does_not_crash(
    hil_driver: Esp32JsonDriver,
) -> None:
    """Driver handles a synthetic ``evt:boot`` frame without crashing.

    The firmware sends a ``boot`` event on power-up.  This test verifies
    the dispatch path does not raise for well-formed ``evt`` frames.
    """
    boot_frame = (
        json.dumps(
            {"t": "evt", "kind": "boot", "ver": "0.0.0-test", "ts": 0.0},
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    hil_driver._dispatch_line(boot_frame)  # type: ignore[attr-defined]
    # Still operational.
    await hil_driver._send_and_await_ack(  # type: ignore[attr-defined]
        cmd="ping",
        payload={},
    )


@pytest.mark.asyncio
async def test_evt_fault_injected_live(
    hil_driver: Esp32JsonDriver,
) -> None:
    """Write a synthetic ``evt:fault`` to the UART and verify no crash.

    Requires ``ARMDROID_HIL_FAULT_INJECT=1``.  Unlike
    ``test_evt_fault_dispatch_does_not_crash``, this actually writes the
    frame to the firmware's input buffer, exercising the reader loop's
    full decode + dispatch path with a live serial connection.

    .. note::

       The firmware will receive and (likely) discard the injected frame
       since it is not a recognised ``cmd`` envelope.  The important
       invariant is that the *host* driver keeps operating.
    """
    if os.environ.get("ARMDROID_HIL_FAULT_INJECT") != "1":
        pytest.skip("ARMDROID_HIL_FAULT_INJECT=1 required for live fault-injection test.")

    fault_frame = (
        json.dumps(
            {"t": "evt", "kind": "fault", "code": "injected", "ts": 0.0},
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    port = hil_driver._port  # type: ignore[attr-defined]
    assert port is not None
    await asyncio.to_thread(port.write, fault_frame)
    await asyncio.to_thread(port.flush)
    await asyncio.sleep(0.15)
    # Driver still alive.
    await hil_driver._send_and_await_ack(  # type: ignore[attr-defined]
        cmd="ping",
        payload={},
    )
