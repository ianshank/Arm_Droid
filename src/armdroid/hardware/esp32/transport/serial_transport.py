"""Serial (UART) transport for the ESP32 JSON driver.

Wraps pyserial in the :class:`ArmTransport` Protocol so the driver does not
import or reference ``serial`` directly.  All blocking pyserial calls run via
:func:`asyncio.to_thread` to keep the event loop unblocked.

Testability
-----------
The ``serial_module`` and ``list_ports_module`` constructor arguments accept
the real ``serial`` / ``serial.tools.list_ports`` module objects, or test
fakes injected by ``monkeypatch``.  The driver passes down whatever it finds
at its own module level, so existing test fixtures that patch
``armdroid.hardware.esp32.driver._serial_module`` continue to work unchanged.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from armdroid.domain.errors import ArmDriverError
from armdroid.hardware.esp32.portfinder import open_port_blocking, resolve_port
from armdroid.logging.setup import get_logger

if TYPE_CHECKING:
    from armdroid.config.schema import ArmConfig

_log = get_logger(__name__)


class SerialTransport:
    """Pyserial-backed :class:`~.base.ArmTransport` implementation."""

    def __init__(
        self,
        cfg: ArmConfig,
        serial_module: Any,
        list_ports_module: Any,
    ) -> None:
        if serial_module is None:
            msg = (
                "pyserial not installed. "
                "Install with `pip install -e .[hardware]` to enable the real-hardware driver."
            )
            raise ArmDriverError(msg)
        self._cfg = cfg
        self._serial_module = serial_module
        self._list_ports_module = list_ports_module
        self._port: Any | None = None

    # ------------------------------------------------------------------ #
    # ArmTransport protocol
    # ------------------------------------------------------------------ #

    @property
    def is_connected(self) -> bool:
        """``True`` if the serial port is currently open."""
        return self._port is not None

    async def connect(self) -> None:
        """Resolve and open the serial port."""
        if self._port is not None:
            return
        port_path = await resolve_port(self._cfg, self._serial_module, self._list_ports_module)
        self._port = await asyncio.to_thread(
            open_port_blocking, port_path, self._cfg, self._serial_module
        )
        _log.info(
            "serial_transport_connected",
            port=port_path,
            baud=self._cfg.transport.serial_baud,
        )

    async def disconnect(self) -> None:
        """Close the serial port.

        Cleanup is best-effort: pyserial occasionally raises on close
        (port already gone, USB unplugged). Such errors are logged at
        DEBUG so they remain visible for forensic analysis.
        """
        port = self._port
        self._port = None
        if port is not None:
            try:
                await asyncio.to_thread(port.close)
            except Exception as exc:  # pragma: no cover - cleanup best-effort
                _log.debug("serial_transport_close_failed", error=str(exc))
            _log.info("serial_transport_disconnected")

    async def write_line(self, data: bytes) -> None:
        """Write *data* to the serial port (blocking, runs in thread pool)."""
        if self._port is None:
            msg = "SerialTransport is not connected"
            raise ArmDriverError(msg)
        await asyncio.to_thread(self._write_blocking, data)

    async def readline(self) -> bytes:
        """Read one newline-terminated line (blocking, runs in thread pool)."""
        if self._port is None:
            return b""
        return await asyncio.to_thread(self._readline_blocking)

    # ------------------------------------------------------------------ #
    # Blocking helpers (executed in asyncio.to_thread)
    # ------------------------------------------------------------------ #

    def _write_blocking(self, data: bytes) -> None:
        if self._port is None:
            msg = "SerialTransport is not connected"
            raise ArmDriverError(msg)
        self._port.write(data)
        self._port.flush()

    def _readline_blocking(self) -> bytes:
        if self._port is None:
            return b""
        result: bytes = self._port.readline()
        return result


__all__ = ["SerialTransport"]
