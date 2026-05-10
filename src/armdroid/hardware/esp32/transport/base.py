r"""ArmTransport Protocol — minimal byte-stream abstraction for ESP32 transports.

Every concrete transport (serial, TCP, BLE) implements this Protocol so the
driver can be written against it without knowing which physical medium is in
use.  The driver owns all protocol semantics (command encoding, pending-reply
futures, keepalive loop, telemetry spans); transports only move bytes.

Wire convention
---------------
All payloads are newline-terminated ASCII/UTF-8 JSON lines, identical to the
serial wire format described in ``firmware/arm_esp32/PROTOCOL.md``.  Each
:meth:`readline` call returns exactly one complete line (including the
trailing ``b"\\n"``), or ``b""`` if no data is available (timeout / EOF).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ArmTransport(Protocol):
    """Minimal byte-stream interface shared by all ESP32 transport backends.

    Implementations must be safe to use from a single asyncio task at a time
    (the driver serialises sends via ``_send_lock`` and runs a single reader
    loop).  Thread-safety across multiple OS threads is *not* required.
    """

    @property
    def is_connected(self) -> bool:
        """``True`` if the transport is currently open and usable."""
        ...

    async def connect(self) -> None:
        """Open the transport.  Idempotent — safe to call when already connected.

        Raises:
            ArmDriverError: If the connection attempt fails.
        """
        ...

    async def disconnect(self) -> None:
        """Close the transport and release all resources.  Idempotent."""
        ...

    async def write_line(self, data: bytes) -> None:
        r"""Write *data* (a complete newline-terminated JSON line) to the peer.

        Args:
            data: Encoded JSON line including the trailing ``b"\\n"``.

        Raises:
            ArmDriverError: If the write fails or the transport is not connected.
        """
        ...

    async def readline(self) -> bytes:
        r"""Read and return the next complete newline-terminated line.

        Returns:
            The raw bytes of the line (including ``b"\\n"``), or ``b""`` on
            timeout / EOF / transport-closed.  Never raises on clean EOF — the
            driver's reader loop uses the empty-bytes sentinel to detect that.
        """
        ...


__all__ = ["ArmTransport"]
