"""TCP (WiFi) transport for the ESP32 JSON driver.

Opens an asyncio TCP connection to the firmware's WiFi server and speaks the
same newline-delimited JSON protocol as serial.  The firmware allows only one
active client at a time; additional connection attempts are rejected until the
current client disconnects.

No third-party dependencies — uses stdlib ``asyncio`` only.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from armdroid.domain.errors import ArmDriverError
from armdroid.logging.setup import get_logger

if TYPE_CHECKING:
    from armdroid.config.schema import ArmConfig

_log = get_logger(__name__)


class TcpTransport:
    """Asyncio TCP client implementing :class:`~.base.ArmTransport`."""

    def __init__(self, cfg: ArmConfig) -> None:
        if cfg.transport.tcp is None:
            msg = "TcpTransport requires transport.tcp config block"
            raise ArmDriverError(msg)
        self._cfg = cfg
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    # ------------------------------------------------------------------ #
    # ArmTransport protocol
    # ------------------------------------------------------------------ #

    @property
    def is_connected(self) -> bool:
        """``True`` if the TCP connection is currently open."""
        return self._writer is not None

    async def connect(self) -> None:
        """Open the TCP connection to the firmware WiFi server."""
        if self._writer is not None:
            return
        tcp = self._cfg.transport.tcp
        assert tcp is not None  # validated in __init__
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(tcp.host, tcp.port),
                timeout=tcp.connect_timeout_s,
            )
        except (TimeoutError, OSError) as exc:
            msg = f"TCP connect to {tcp.host}:{tcp.port} failed: {exc}"
            raise ArmDriverError(msg) from exc
        self._reader = reader
        self._writer = writer
        _log.info(
            "tcp_transport_connected",
            host=tcp.host,
            port=tcp.port,
        )

    async def disconnect(self) -> None:
        """Close the TCP connection."""
        writer = self._writer
        self._writer = None
        self._reader = None
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass
            _log.info("tcp_transport_disconnected")

    async def write_line(self, data: bytes) -> None:
        """Write *data* to the TCP connection."""
        if self._writer is None:
            msg = "TcpTransport is not connected"
            raise ArmDriverError(msg)
        try:
            self._writer.write(data)
            await self._writer.drain()
        except OSError as exc:
            msg = f"TCP write failed: {exc}"
            raise ArmDriverError(msg) from exc

    async def readline(self) -> bytes:
        """Read one newline-terminated line from the TCP connection.

        Returns ``b""`` on EOF or closed connection.  When the connection is
        broken mid-read the internal state is reset so :attr:`is_connected`
        immediately reflects the loss and the driver can react.
        """
        if self._reader is None:
            return b""
        try:
            line = await self._reader.readline()
            return line
        except (ConnectionResetError, OSError) as exc:
            # Connection lost — clear state so is_connected => False and
            # the driver's reader loop exits cleanly rather than spinning.
            self._writer = None
            self._reader = None
            _log.warning("tcp_transport_connection_lost", error=str(exc))
            return b""


__all__ = ["TcpTransport"]
