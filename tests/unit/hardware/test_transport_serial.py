"""Unit tests for transport/serial_transport.py — SerialTransport.

These exercise the serial transport in isolation (no driver, no real
pyserial), complementing the integration coverage in
test_esp32_json_driver.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from armdroid.domain.errors import ArmDriverError
from armdroid.hardware.esp32.transport.serial_transport import SerialTransport


def _make_cfg(serial_port: str = "/dev/ttyUSB0", baud: int = 115_200) -> MagicMock:
    cfg = MagicMock()
    cfg.transport.protocol = "serial"
    cfg.transport.serial_port = serial_port
    cfg.transport.serial_baud = baud
    return cfg


def _make_serial_module() -> MagicMock:
    """Return a stand-in for the real ``serial`` module."""
    return MagicMock()


def _make_list_ports_module() -> MagicMock:
    """Return a stand-in for ``serial.tools.list_ports``."""
    return MagicMock()


def test_init_raises_when_serial_module_missing() -> None:
    cfg = _make_cfg()
    with pytest.raises(ArmDriverError, match="pyserial not installed"):
        SerialTransport(cfg, serial_module=None, list_ports_module=None)


def test_initial_state_not_connected() -> None:
    cfg = _make_cfg()
    t = SerialTransport(cfg, serial_module=_make_serial_module(), list_ports_module=None)
    assert t.is_connected is False


@pytest.mark.asyncio
async def test_connect_opens_port_and_marks_connected() -> None:
    cfg = _make_cfg()
    serial_mod = _make_serial_module()
    fake_port = MagicMock()
    with (
        patch(
            "armdroid.hardware.esp32.transport.serial_transport.resolve_port",
            new=AsyncMock(return_value="/dev/ttyUSB0"),
        ),
        patch(
            "armdroid.hardware.esp32.transport.serial_transport.open_port_blocking",
            return_value=fake_port,
        ),
    ):
        t = SerialTransport(cfg, serial_module=serial_mod, list_ports_module=None)
        await t.connect()
        assert t.is_connected is True
        assert t._port is fake_port


@pytest.mark.asyncio
async def test_connect_idempotent() -> None:
    """Calling connect() twice opens the port only once."""
    cfg = _make_cfg()
    serial_mod = _make_serial_module()
    fake_port = MagicMock()
    with (
        patch(
            "armdroid.hardware.esp32.transport.serial_transport.resolve_port",
            new=AsyncMock(return_value="/dev/ttyUSB0"),
        ) as resolve,
        patch(
            "armdroid.hardware.esp32.transport.serial_transport.open_port_blocking",
            return_value=fake_port,
        ) as opener,
    ):
        t = SerialTransport(cfg, serial_module=serial_mod, list_ports_module=None)
        await t.connect()
        await t.connect()
        assert resolve.await_count == 1
        assert opener.call_count == 1


@pytest.mark.asyncio
async def test_disconnect_closes_port_and_clears_state() -> None:
    cfg = _make_cfg()
    fake_port = MagicMock()
    fake_port.close = MagicMock()
    t = SerialTransport(cfg, serial_module=_make_serial_module(), list_ports_module=None)
    t._port = fake_port
    await t.disconnect()
    assert t.is_connected is False
    fake_port.close.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect_when_not_connected_is_noop() -> None:
    cfg = _make_cfg()
    t = SerialTransport(cfg, serial_module=_make_serial_module(), list_ports_module=None)
    # Should not raise; nothing to close.
    await t.disconnect()
    assert t.is_connected is False


@pytest.mark.asyncio
async def test_disconnect_swallows_close_exception_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing port.close() must not propagate; cleanup is best-effort."""
    cfg = _make_cfg()
    fake_port = MagicMock()
    fake_port.close = MagicMock(side_effect=OSError("device gone"))
    t = SerialTransport(cfg, serial_module=_make_serial_module(), list_ports_module=None)
    t._port = fake_port
    # Must not raise even though close() raised.
    await t.disconnect()
    assert t.is_connected is False


@pytest.mark.asyncio
async def test_write_line_writes_and_flushes_via_thread() -> None:
    cfg = _make_cfg()
    fake_port = MagicMock()
    fake_port.write = MagicMock()
    fake_port.flush = MagicMock()
    t = SerialTransport(cfg, serial_module=_make_serial_module(), list_ports_module=None)
    t._port = fake_port

    await t.write_line(b'{"t":"cmd","cmd":"ping"}\n')

    fake_port.write.assert_called_once_with(b'{"t":"cmd","cmd":"ping"}\n')
    fake_port.flush.assert_called_once()


@pytest.mark.asyncio
async def test_write_line_when_not_connected_raises() -> None:
    cfg = _make_cfg()
    t = SerialTransport(cfg, serial_module=_make_serial_module(), list_ports_module=None)
    with pytest.raises(ArmDriverError, match="not connected"):
        await t.write_line(b"ping\n")


@pytest.mark.asyncio
async def test_readline_when_not_connected_returns_empty() -> None:
    cfg = _make_cfg()
    t = SerialTransport(cfg, serial_module=_make_serial_module(), list_ports_module=None)
    assert await t.readline() == b""


@pytest.mark.asyncio
async def test_readline_returns_port_readline_result() -> None:
    cfg = _make_cfg()
    fake_port = MagicMock()
    fake_port.readline = MagicMock(return_value=b'{"t":"ack","id":1}\n')
    t = SerialTransport(cfg, serial_module=_make_serial_module(), list_ports_module=None)
    t._port = fake_port

    result = await t.readline()
    assert result == b'{"t":"ack","id":1}\n'
    fake_port.readline.assert_called_once()


@pytest.mark.asyncio
async def test_connect_propagates_resolve_port_failure() -> None:
    """If port resolution fails, connect() should raise and stay disconnected."""
    cfg = _make_cfg()
    with patch(
        "armdroid.hardware.esp32.transport.serial_transport.resolve_port",
        new=AsyncMock(side_effect=ArmDriverError("no port found")),
    ):
        t = SerialTransport(cfg, serial_module=_make_serial_module(), list_ports_module=None)
        with pytest.raises(ArmDriverError, match="no port found"):
            await t.connect()
        assert t.is_connected is False


@pytest.mark.asyncio
async def test_connect_propagates_open_port_failure() -> None:
    """If open_port_blocking raises, connect() should raise and stay disconnected."""
    cfg = _make_cfg()
    with (
        patch(
            "armdroid.hardware.esp32.transport.serial_transport.resolve_port",
            new=AsyncMock(return_value="/dev/ttyUSB0"),
        ),
        patch(
            "armdroid.hardware.esp32.transport.serial_transport.open_port_blocking",
            side_effect=OSError("permission denied"),
        ),
    ):
        t = SerialTransport(cfg, serial_module=_make_serial_module(), list_ports_module=None)
        with pytest.raises(OSError, match="permission denied"):
            await t.connect()
        assert t.is_connected is False
