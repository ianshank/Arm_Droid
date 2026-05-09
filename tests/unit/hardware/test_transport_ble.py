"""Unit tests for transport/ble_transport.py — BleTransport with fake client injection."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from armdroid.domain.errors import ArmDriverError
from armdroid.hardware.esp32.transport.ble_transport import BleTransport

# ---------------------------------------------------------------------------
# Fake BleakClient for injection
# ---------------------------------------------------------------------------


class FakeBleClient:
    """Minimal fake BleakClient implementing BleakClientProtocol."""

    def __init__(self, address: str) -> None:
        self.address = address
        self._connected = False
        self._notify_cb: Callable[[Any, bytearray], None] | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        self._connected = True
        return True

    async def disconnect(self) -> bool:
        self._connected = False
        return True

    async def write_gatt_char(
        self,
        char_specifier: str,
        data: bytes | bytearray,
        response: bool = True,
    ) -> None:
        self.last_written = (char_specifier, bytes(data))

    async def start_notify(
        self,
        char_specifier: str,
        callback: Callable[[Any, bytearray], None],
    ) -> None:
        self._notify_cb = callback

    async def stop_notify(self, char_specifier: str) -> None:
        self._notify_cb = None

    def inject_notification(self, data: bytes) -> None:
        """Simulate the firmware sending a BLE notification."""
        if self._notify_cb is not None:
            self._notify_cb(self, bytearray(data))


def _make_cfg(address: str = "AA:BB:CC:DD:EE:FF") -> MagicMock:
    cfg = MagicMock()
    cfg.transport.protocol = "ble"
    cfg.transport.ble = MagicMock()
    cfg.transport.ble.device_address = address
    cfg.transport.ble.device_name = "ArmDroid"
    cfg.transport.ble.rx_char_uuid = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
    cfg.transport.ble.tx_char_uuid = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
    cfg.transport.ble.scan_timeout_s = 5.0
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ble_transport_connect_success() -> None:
    cfg = _make_cfg()
    fake_client = FakeBleClient("AA:BB:CC:DD:EE:FF")
    t = BleTransport(cfg, client_factory=lambda addr: fake_client)
    assert not t.is_connected
    await t.connect()
    assert t.is_connected
    assert fake_client.is_connected


@pytest.mark.asyncio
async def test_ble_transport_connect_idempotent() -> None:
    cfg = _make_cfg()
    fake_client = FakeBleClient("AA:BB:CC:DD:EE:FF")
    calls = 0
    original_connect = fake_client.connect

    async def counting_connect() -> bool:
        nonlocal calls
        calls += 1
        return await original_connect()

    fake_client.connect = counting_connect  # type: ignore[method-assign]
    t = BleTransport(cfg, client_factory=lambda addr: fake_client)
    await t.connect()
    await t.connect()
    assert calls == 1


@pytest.mark.asyncio
async def test_ble_transport_disconnect() -> None:
    cfg = _make_cfg()
    fake_client = FakeBleClient("AA:BB:CC:DD:EE:FF")
    t = BleTransport(cfg, client_factory=lambda addr: fake_client)
    await t.connect()
    await t.disconnect()
    assert not t.is_connected


@pytest.mark.asyncio
async def test_ble_transport_readline_receives_notification() -> None:
    cfg = _make_cfg()
    fake_client = FakeBleClient("AA:BB:CC:DD:EE:FF")
    t = BleTransport(cfg, client_factory=lambda addr: fake_client)
    await t.connect()

    # Inject a complete newline-terminated notification line
    line = b'{"t":"state","q":[0,0,0,0,0,0],"qd":[0,0,0,0,0,0]}\n'
    fake_client.inject_notification(line)

    result = await t.readline()
    assert result == line


@pytest.mark.asyncio
async def test_ble_transport_readline_reassembles_fragments() -> None:
    """Fragmented notifications are reassembled into complete lines."""
    cfg = _make_cfg()
    fake_client = FakeBleClient("AA:BB:CC:DD:EE:FF")
    t = BleTransport(cfg, client_factory=lambda addr: fake_client)
    await t.connect()

    full_line = b'{"t":"ack","id":1}\n'
    half = len(full_line) // 2
    fake_client.inject_notification(full_line[:half])
    fake_client.inject_notification(full_line[half:])

    result = await t.readline()
    assert result == full_line


@pytest.mark.asyncio
async def test_ble_transport_readline_multiple_lines_in_one_notification() -> None:
    """Multiple newline-delimited lines in a single notification are both enqueued."""
    cfg = _make_cfg()
    fake_client = FakeBleClient("AA:BB:CC:DD:EE:FF")
    t = BleTransport(cfg, client_factory=lambda addr: fake_client)
    await t.connect()

    line1 = b'{"t":"ack","id":1}\n'
    line2 = b'{"t":"ack","id":2}\n'
    fake_client.inject_notification(line1 + line2)

    r1 = await t.readline()
    r2 = await t.readline()
    assert r1 == line1
    assert r2 == line2


@pytest.mark.asyncio
async def test_ble_transport_write_gatt_char() -> None:
    cfg = _make_cfg()
    fake_client = FakeBleClient("AA:BB:CC:DD:EE:FF")
    t = BleTransport(cfg, client_factory=lambda addr: fake_client)
    await t.connect()

    data = b'{"t":"cmd","cmd":"ping"}\n'
    await t.write_line(data)
    assert fake_client.last_written == (cfg.transport.ble.rx_char_uuid, data)


@pytest.mark.asyncio
async def test_ble_transport_write_not_connected_raises() -> None:
    cfg = _make_cfg()
    t = BleTransport(cfg, client_factory=lambda addr: FakeBleClient(addr))
    with pytest.raises(ArmDriverError, match="not connected"):
        await t.write_line(b"ping\n")


@pytest.mark.asyncio
async def test_ble_transport_disconnect_unblocks_readline() -> None:
    """readline() returns b"" after disconnect (sentinel unblocks awaiter)."""
    cfg = _make_cfg()
    fake_client = FakeBleClient("AA:BB:CC:DD:EE:FF")
    t = BleTransport(cfg, client_factory=lambda addr: fake_client)
    await t.connect()

    # Start a readline that will block, then disconnect.
    read_task = asyncio.create_task(t.readline())
    await asyncio.sleep(0)  # yield to let task reach the queue.get()
    await t.disconnect()
    result = await read_task
    assert result == b""


def test_ble_transport_requires_ble_config() -> None:
    cfg = MagicMock()
    cfg.transport.ble = None
    with pytest.raises(ArmDriverError, match="ble config"):
        BleTransport(cfg)


@pytest.mark.asyncio
async def test_ble_transport_no_bleak_without_factory_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without bleak installed and no factory, connect() raises ArmDriverError."""
    cfg = _make_cfg()
    t = BleTransport(cfg, client_factory=None)
    # Patch _bleak_available to return False
    import armdroid.hardware.esp32.transport.ble_transport as _mod

    monkeypatch.setattr(_mod, "_bleak_available", lambda: False)
    with pytest.raises(ArmDriverError, match="bleak not installed"):
        await t.connect()


@pytest.mark.asyncio
async def test_ble_transport_unexpected_disconnect_clears_is_connected() -> None:
    """_on_ble_disconnect sets is_connected=False and unblocks readline."""
    cfg = _make_cfg()
    fake_client = FakeBleClient("AA:BB:CC:DD:EE:FF")
    t = BleTransport(cfg, client_factory=lambda addr: fake_client)
    await t.connect()
    assert t.is_connected

    # Simulate bleak calling the disconnected_callback from a thread
    # (in tests we drive it synchronously on the event loop via _handle_unexpected_disconnect).
    t._handle_unexpected_disconnect()

    assert not t.is_connected
    # The sentinel must have been queued so any awaiting readline returns b"".
    result = t._rx_queue.get_nowait()
    assert result == b""


@pytest.mark.asyncio
async def test_ble_transport_buffer_overflow_protection() -> None:
    """_ingest_chunk discards the buffer when it exceeds _MAX_NOTIFY_BYTES without a newline."""
    cfg = _make_cfg()
    fake_client = FakeBleClient("AA:BB:CC:DD:EE:FF")
    t = BleTransport(cfg, client_factory=lambda addr: fake_client)
    await t.connect()

    # Inject a chunk larger than _MAX_NOTIFY_BYTES with no newline.
    garbage = b"x" * (BleTransport._MAX_NOTIFY_BYTES + 1)
    t._ingest_chunk(garbage)

    # Buffer must have been cleared — no items in queue.
    assert t._rx_queue.empty()
    assert len(t._rx_buf) == 0


# ---------------------------------------------------------------------------
# _scan_for_device tests — exercise the auto-discovery path
# ---------------------------------------------------------------------------


def _install_fake_bleak(
    monkeypatch: pytest.MonkeyPatch,
    *,
    find_result: Any,
    raise_timeout: bool = False,
) -> None:
    """Install a fake ``bleak`` module exposing ``BleakScanner.find_device_by_name``.

    Use ``find_result`` to control the returned device (or None).  Set
    ``raise_timeout=True`` to make ``asyncio.wait_for`` see a coroutine
    that never completes within the configured timeout.
    """
    import sys

    fake_bleak = MagicMock()

    if raise_timeout:

        async def _slow_find(_name: str) -> Any:
            await asyncio.sleep(10)  # longer than scan_timeout_s
            return None

        fake_bleak.BleakScanner.find_device_by_name = _slow_find
    else:

        async def _find(_name: str) -> Any:
            return find_result

        fake_bleak.BleakScanner.find_device_by_name = _find

    monkeypatch.setitem(sys.modules, "bleak", fake_bleak)


@pytest.mark.asyncio
async def test_scan_for_device_returns_address_when_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When BleakScanner finds a device by name, its address is returned."""
    cfg = _make_cfg()
    cfg.transport.ble.scan_timeout_s = 1.0

    fake_device = MagicMock()
    fake_device.address = "11:22:33:44:55:66"
    _install_fake_bleak(monkeypatch, find_result=fake_device)

    fake_client = FakeBleClient("11:22:33:44:55:66")
    t = BleTransport(cfg, client_factory=lambda addr: fake_client)
    address = await t._scan_for_device(cfg.transport.ble)
    assert address == "11:22:33:44:55:66"


@pytest.mark.asyncio
async def test_scan_for_device_raises_when_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the device is not discovered, ArmDriverError is raised."""
    cfg = _make_cfg()
    cfg.transport.ble.scan_timeout_s = 1.0
    _install_fake_bleak(monkeypatch, find_result=None)

    t = BleTransport(cfg, client_factory=lambda addr: FakeBleClient(addr))
    with pytest.raises(ArmDriverError, match="not found"):
        await t._scan_for_device(cfg.transport.ble)


@pytest.mark.asyncio
async def test_scan_for_device_raises_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scan that exceeds scan_timeout_s surfaces as ArmDriverError."""
    cfg = _make_cfg()
    cfg.transport.ble.scan_timeout_s = 0.01  # very short timeout
    _install_fake_bleak(monkeypatch, find_result=None, raise_timeout=True)

    t = BleTransport(cfg, client_factory=lambda addr: FakeBleClient(addr))
    with pytest.raises(ArmDriverError, match="timed out"):
        await t._scan_for_device(cfg.transport.ble)


@pytest.mark.asyncio
async def test_connect_with_auto_address_invokes_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """device_address='auto' triggers _scan_for_device on connect()."""
    cfg = _make_cfg(address="auto")
    cfg.transport.ble.scan_timeout_s = 1.0

    fake_device = MagicMock()
    fake_device.address = "DE:AD:BE:EF:00:01"
    _install_fake_bleak(monkeypatch, find_result=fake_device)
    # Force _bleak_available True so connect() proceeds without a real bleak install.
    import armdroid.hardware.esp32.transport.ble_transport as _mod

    monkeypatch.setattr(_mod, "_bleak_available", lambda: True)

    fake_client = FakeBleClient("DE:AD:BE:EF:00:01")
    received: list[str] = []

    def _factory(addr: str) -> FakeBleClient:
        received.append(addr)
        return fake_client

    t = BleTransport(cfg, client_factory=_factory)
    await t.connect()
    assert received == ["DE:AD:BE:EF:00:01"]
    assert t.is_connected
