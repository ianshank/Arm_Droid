"""Unit tests for transport/tcp_transport.py — TcpTransport."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from armdroid.domain.errors import ArmDriverError
from armdroid.hardware.esp32.transport.tcp_transport import TcpTransport


def _make_cfg(host: str = "127.0.0.1", port: int = 3001, timeout: float = 2.0) -> MagicMock:
    cfg = MagicMock()
    cfg.transport.protocol = "tcp"
    cfg.transport.tcp = MagicMock()
    cfg.transport.tcp.host = host
    cfg.transport.tcp.port = port
    cfg.transport.tcp.connect_timeout_s = timeout
    return cfg


def _make_stream_pair(lines: list[bytes] | None = None) -> tuple[MagicMock, MagicMock]:
    reader = MagicMock()
    writer = MagicMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    if lines is not None:
        reader.readline = AsyncMock(side_effect=lines)
    else:
        reader.readline = AsyncMock(return_value=b'{"t":"ack","id":1}\n')
    return reader, writer


@pytest.mark.asyncio
async def test_tcp_transport_connect_success() -> None:
    cfg = _make_cfg()
    reader, writer = _make_stream_pair()
    with patch("asyncio.open_connection", AsyncMock(return_value=(reader, writer))):
        t = TcpTransport(cfg)
        assert not t.is_connected
        await t.connect()
        assert t.is_connected


@pytest.mark.asyncio
async def test_tcp_transport_connect_idempotent() -> None:
    cfg = _make_cfg()
    reader, writer = _make_stream_pair()
    with patch("asyncio.open_connection", AsyncMock(return_value=(reader, writer))) as mock_conn:
        t = TcpTransport(cfg)
        await t.connect()
        await t.connect()
        mock_conn.assert_called_once()


@pytest.mark.asyncio
async def test_tcp_transport_connect_timeout_raises() -> None:
    cfg = _make_cfg(timeout=0.01)
    with patch("asyncio.open_connection", AsyncMock(side_effect=TimeoutError)):
        t = TcpTransport(cfg)
        with pytest.raises(ArmDriverError, match="connect"):
            await t.connect()


@pytest.mark.asyncio
async def test_tcp_transport_connect_os_error_raises() -> None:
    cfg = _make_cfg()
    with patch("asyncio.open_connection", AsyncMock(side_effect=OSError("refused"))):
        t = TcpTransport(cfg)
        with pytest.raises(ArmDriverError, match="refused"):
            await t.connect()


@pytest.mark.asyncio
async def test_tcp_transport_disconnect() -> None:
    cfg = _make_cfg()
    reader, writer = _make_stream_pair()
    with patch("asyncio.open_connection", AsyncMock(return_value=(reader, writer))):
        t = TcpTransport(cfg)
        await t.connect()
        await t.disconnect()
        assert not t.is_connected
        writer.close.assert_called_once()
        writer.wait_closed.assert_called_once()


@pytest.mark.asyncio
async def test_tcp_transport_readline_returns_line() -> None:
    cfg = _make_cfg()
    expected = b'{"t":"state"}\n'
    reader, writer = _make_stream_pair(lines=[expected])
    with patch("asyncio.open_connection", AsyncMock(return_value=(reader, writer))):
        t = TcpTransport(cfg)
        await t.connect()
        result = await t.readline()
        assert result == expected


@pytest.mark.asyncio
async def test_tcp_transport_readline_returns_empty_on_eof() -> None:
    cfg = _make_cfg()
    reader, writer = _make_stream_pair(lines=[b""])
    with patch("asyncio.open_connection", AsyncMock(return_value=(reader, writer))):
        t = TcpTransport(cfg)
        await t.connect()
        result = await t.readline()
        assert result == b""


@pytest.mark.asyncio
async def test_tcp_transport_readline_when_not_connected_returns_empty() -> None:
    cfg = _make_cfg()
    t = TcpTransport(cfg)
    result = await t.readline()
    assert result == b""


@pytest.mark.asyncio
async def test_tcp_transport_write_line_sends_data() -> None:
    cfg = _make_cfg()
    reader, writer = _make_stream_pair()
    with patch("asyncio.open_connection", AsyncMock(return_value=(reader, writer))):
        t = TcpTransport(cfg)
        await t.connect()
        data = b'{"t":"cmd","cmd":"ping"}\n'
        await t.write_line(data)
        writer.write.assert_called_once_with(data)
        writer.drain.assert_called_once()


@pytest.mark.asyncio
async def test_tcp_transport_write_line_not_connected_raises() -> None:
    cfg = _make_cfg()
    t = TcpTransport(cfg)
    with pytest.raises(ArmDriverError, match="not connected"):
        await t.write_line(b"ping\n")


@pytest.mark.asyncio
async def test_tcp_transport_readline_clears_state_on_os_error() -> None:
    """After an OSError on readline, is_connected must become False."""
    cfg = _make_cfg()
    reader, writer = _make_stream_pair(lines=[OSError("connection reset")])
    reader.readline = AsyncMock(side_effect=OSError("connection reset"))
    with patch("asyncio.open_connection", AsyncMock(return_value=(reader, writer))):
        t = TcpTransport(cfg)
        await t.connect()
        assert t.is_connected
        result = await t.readline()
        assert result == b""
        # Transport must self-report as disconnected so the driver can react.
        assert not t.is_connected


@pytest.mark.asyncio
async def test_tcp_transport_is_connected_false_allows_reconnect() -> None:
    """After a connection-lost readline, connect() opens a fresh connection."""
    cfg = _make_cfg()
    reader1, writer1 = _make_stream_pair(lines=[OSError("reset")])
    reader1.readline = AsyncMock(side_effect=OSError("reset"))
    reader2, writer2 = _make_stream_pair()
    call_count = 0

    async def open_conn(*args: object, **kwargs: object) -> tuple[MagicMock, MagicMock]:
        nonlocal call_count
        call_count += 1
        return (reader1, writer1) if call_count == 1 else (reader2, writer2)

    with patch("asyncio.open_connection", side_effect=open_conn):
        t = TcpTransport(cfg)
        await t.connect()
        await t.readline()  # triggers OSError, clears state
        assert not t.is_connected
        await t.connect()  # must re-open
        assert t.is_connected
        assert call_count == 2


def test_tcp_transport_requires_tcp_config() -> None:
    cfg = MagicMock()
    cfg.transport.tcp = None
    with pytest.raises(ArmDriverError, match="tcp config"):
        TcpTransport(cfg)
