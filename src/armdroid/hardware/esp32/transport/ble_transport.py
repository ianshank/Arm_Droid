"""BLE GATT transport for the ESP32 JSON driver.

Speaks the same newline-delimited JSON protocol as serial/TCP over two BLE
GATT characteristics (NUS-style):

* **RX char** (host → device): host *writes* JSON command lines.
* **TX char** (device → host): device *notifies* state, ack, nak, evt lines.

Notification data is buffered in an :class:`asyncio.Queue` and consumed by
:meth:`readline`, preserving the same pull-based interface the driver expects.

``bleak`` is a soft dependency (``[ble]`` extra).  Importing this module on a
system without ``bleak`` is fine; the error is raised only on
:meth:`BleTransport.connect`.

Testability
-----------
Pass a ``client_factory`` callable to :class:`BleTransport` to inject a fake
:class:`BleakClientProtocol` in tests, avoiding any real BLE stack.  The
factory receives the resolved device address as its only argument.

    factory: BleakClientFactory = lambda addr: MyFakeBleClient(addr)
    transport = BleTransport(cfg, client_factory=factory)
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from armdroid.domain.errors import ArmDriverError
from armdroid.logging.setup import get_logger

if TYPE_CHECKING:
    from armdroid.config.schema import ArmConfig, BleTransportConfig

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# BleakClientProtocol — structural type for injection / testing
# ---------------------------------------------------------------------------


@runtime_checkable
class BleakClientProtocol(Protocol):
    """Structural Protocol mirroring the ``BleakClient`` interface.

    Concrete ``BleakClient`` instances satisfy this Protocol when ``bleak``
    is installed.  Pass a factory that returns a fake implementation in tests
    to avoid importing ``bleak`` or touching a BLE stack.
    """

    @property
    def is_connected(self) -> bool:
        """Whether the client is currently connected."""
        ...

    async def connect(self) -> bool | None:
        """Connect to the BLE device.  Returns True on success (bleak 0.22+)."""
        ...

    async def disconnect(self) -> bool | None:
        """Disconnect from the BLE device."""
        ...

    async def write_gatt_char(
        self,
        char_specifier: str,
        data: bytes | bytearray,
        response: bool = True,
    ) -> None:
        """Write *data* to a GATT characteristic."""
        ...

    async def start_notify(
        self,
        char_specifier: str,
        callback: Callable[[Any, bytearray], None],
    ) -> None:
        """Subscribe to notifications on a GATT characteristic."""
        ...

    async def stop_notify(self, char_specifier: str) -> None:
        """Unsubscribe from notifications on a GATT characteristic."""
        ...


#: Factory callable type: ``(device_address: str) -> BleakClientProtocol``
BleakClientFactory = Callable[[str], BleakClientProtocol]


def _bleak_available() -> bool:
    """Return ``True`` if the ``bleak`` package can be imported."""
    from importlib.util import find_spec

    return find_spec("bleak") is not None


# ---------------------------------------------------------------------------
# BleTransport
# ---------------------------------------------------------------------------


class BleTransport:
    """BLE GATT client implementing :class:`~.base.ArmTransport`.

    Args:
        cfg: Arm configuration.  ``cfg.transport.ble`` must be populated.
        client_factory: Optional factory for injecting a fake
            :class:`BleakClientProtocol` in tests.  When ``None`` (default),
            a real :class:`bleak.BleakClient` is constructed.
    """

    #: Hard floor for the in-memory notification reassembly buffer. The
    #: NUS TX characteristic MTU is typically 20 bytes; one JSON line
    #: comfortably fits in well under 512 bytes. The actual cap used at
    #: runtime is ``max(_MIN_NOTIFY_BUF_BYTES, cfg.transport.max_line_bytes)``
    #: so operators raising ``max_line_bytes`` for larger frames cannot
    #: have their frames silently dropped here. The floor protects against
    #: a misconfigured ``max_line_bytes=0`` from disabling reassembly
    #: entirely.
    _MIN_NOTIFY_BUF_BYTES: int = 512

    def __init__(
        self,
        cfg: ArmConfig,
        client_factory: BleakClientFactory | None = None,
    ) -> None:
        if cfg.transport.ble is None:
            msg = "BleTransport requires transport.ble config block"
            raise ArmDriverError(msg)
        self._cfg = cfg
        self._client_factory = client_factory
        self._client: BleakClientProtocol | None = None
        # Queue populated by the BLE notification callback.
        self._rx_queue: asyncio.Queue[bytes] = asyncio.Queue()
        # Reassembly buffer for fragmented notification chunks.
        self._rx_buf: bytearray = bytearray()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._is_connected: bool = False
        # Resolve the reassembly cap once at init. ``cfg`` is sometimes a
        # ``MagicMock`` in tests; comparing an int with a MagicMock
        # cascades truthy and would silently corrupt the overflow check
        # at every notification. Coercing to ``int`` here pins the cap
        # to a real number for the lifetime of the transport.
        try:
            cfg_max = int(cfg.transport.max_line_bytes)
        except (TypeError, ValueError):
            cfg_max = self._MIN_NOTIFY_BUF_BYTES
        self._rx_buf_max_bytes: int = max(self._MIN_NOTIFY_BUF_BYTES, cfg_max)

    # ------------------------------------------------------------------ #
    # ArmTransport protocol
    # ------------------------------------------------------------------ #

    @property
    def is_connected(self) -> bool:
        """``True`` if the BLE client is currently connected."""
        return self._is_connected

    async def connect(self) -> None:
        """Scan (if needed), connect, and subscribe to TX notifications."""
        if self._is_connected:
            return
        if not _bleak_available() and self._client_factory is None:
            msg = (
                "bleak not installed. "
                "Install with `pip install -e .[ble]` to enable the BLE transport."
            )
            raise ArmDriverError(msg)

        # Drain stale RX state from any previous connection, including the b""
        # sentinel that disconnect() / _handle_unexpected_disconnect() enqueue.
        # Without this, a reconnect can immediately see phantom empty reads or
        # deliver pre-disconnect frame fragments to the new session.
        self._drain_rx_state()

        self._loop = asyncio.get_running_loop()
        ble: BleTransportConfig = self._cfg.transport.ble  # type: ignore[assignment]

        address = ble.device_address
        if address == "auto":
            address = await self._scan_for_device(ble)

        if self._client_factory is not None:
            client = self._client_factory(address)
        else:
            from bleak import BleakClient

            client = BleakClient(address, disconnected_callback=self._on_ble_disconnect)

        await client.connect()
        self._client = client

        # Subscribe to device→host notifications.
        # The callback runs in the event loop on most bleak backends;
        # call_soon_threadsafe guards the WinRT / BlueZ thread-hop case.
        loop = self._loop

        def _notify_cb(sender: Any, data: bytearray) -> None:
            loop.call_soon_threadsafe(self._ingest_chunk, bytes(data))

        await client.start_notify(ble.tx_char_uuid, _notify_cb)
        self._is_connected = True
        _log.info("ble_transport_connected", address=address)

    async def disconnect(self) -> None:
        """Stop notifications and disconnect.

        Cleanup is best-effort: bleak occasionally raises on teardown
        (peripheral already gone, BlueZ DBus glitch). Such errors are
        logged at DEBUG so they remain visible without scaring callers.
        """
        self._is_connected = False
        client = self._client
        self._client = None
        if client is not None:
            ble: BleTransportConfig = self._cfg.transport.ble  # type: ignore[assignment]
            try:
                await client.stop_notify(ble.tx_char_uuid)
            except Exception as exc:  # pragma: no cover - cleanup best-effort
                _log.debug("ble_transport_stop_notify_failed", error=str(exc))
            try:
                await client.disconnect()
            except Exception as exc:  # pragma: no cover - cleanup best-effort
                _log.debug("ble_transport_client_disconnect_failed", error=str(exc))
            _log.info("ble_transport_disconnected")
        # Unblock any pending readline() awaits.
        await self._rx_queue.put(b"")

    async def write_line(self, data: bytes) -> None:
        """Write *data* to the host→device RX characteristic."""
        if self._client is None:
            msg = "BleTransport is not connected"
            raise ArmDriverError(msg)
        ble: BleTransportConfig = self._cfg.transport.ble  # type: ignore[assignment]
        try:
            await self._client.write_gatt_char(ble.rx_char_uuid, data, response=True)
        except Exception as exc:
            msg = f"BLE write failed: {exc}"
            raise ArmDriverError(msg) from exc

    async def readline(self) -> bytes:
        """Return the next complete newline-terminated line from the RX queue.

        Blocks until a full line is available or the transport disconnects
        (in which case ``b""`` is returned).
        """
        return await self._rx_queue.get()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _ingest_chunk(self, chunk: bytes) -> None:
        """Reassemble notification chunks into complete lines.

        If the buffer grows beyond the configured maximum line length
        without a newline (e.g. firmware bug or a corrupted stream) the
        buffer is discarded to prevent unbounded memory growth. The cap
        is sourced from ``cfg.transport.max_line_bytes`` (with a hard
        floor of :attr:`_MIN_NOTIFY_BUF_BYTES`) so operators tuning
        max_line_bytes upward also lift the BLE reassembly cap.
        """
        self._rx_buf.extend(chunk)
        if len(self._rx_buf) > self._rx_buf_max_bytes and b"\n" not in self._rx_buf:
            _log.warning(
                "ble_transport_rx_buffer_overflow",
                buf_len=len(self._rx_buf),
                max_bytes=self._rx_buf_max_bytes,
            )
            self._rx_buf.clear()
            return
        # Flush all complete lines from the buffer.
        while b"\n" in self._rx_buf:
            idx = self._rx_buf.index(b"\n")
            line = bytes(self._rx_buf[: idx + 1])
            self._rx_buf = self._rx_buf[idx + 1 :]
            self._rx_queue.put_nowait(line)

    def _on_ble_disconnect(self, _client: Any) -> None:
        """Called by bleak when the peripheral disconnects unexpectedly."""
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._handle_unexpected_disconnect)

    def _handle_unexpected_disconnect(self) -> None:
        """Run on the event loop when bleak fires the disconnected callback."""
        self._is_connected = False
        # Clear the reassembly buffer; any in-flight partial frame belongs to
        # the now-dead connection and must not be delivered to a future session.
        self._rx_buf.clear()
        self._rx_queue.put_nowait(b"")
        _log.warning("ble_transport_unexpected_disconnect")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _drain_rx_state(self) -> None:
        """Drain the RX queue and clear the reassembly buffer.

        Called at the start of :meth:`connect` to discard any stale sentinels
        or leftover frame fragments from a previous connection so the new
        session starts with a clean slate.
        """
        while not self._rx_queue.empty():
            try:
                self._rx_queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - race-safe guard
                break
        self._rx_buf.clear()

    async def _scan_for_device(self, ble: BleTransportConfig) -> str:
        """Scan for a BLE device matching ``ble.device_name``."""
        from bleak import BleakScanner

        name = ble.device_name or "ArmDroid"
        _log.info("ble_transport_scanning", name=name, timeout_s=ble.scan_timeout_s)
        try:
            device = await asyncio.wait_for(
                BleakScanner.find_device_by_name(name),
                timeout=ble.scan_timeout_s,
            )
        except TimeoutError as exc:
            msg = f"BLE scan timed out after {ble.scan_timeout_s}s"
            raise ArmDriverError(msg) from exc
        if device is None:
            msg = f"BLE device named {name!r} not found"
            raise ArmDriverError(msg)
        address: str = device.address
        _log.info("ble_transport_device_found", name=name, address=address)
        return address


__all__ = ["BleTransport", "BleakClientFactory", "BleakClientProtocol"]
