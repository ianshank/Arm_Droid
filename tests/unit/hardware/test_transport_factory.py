"""Unit tests for transport/factory.py — make_transport factory."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from armdroid.hardware.esp32.transport.auth import AuthTransport
from armdroid.hardware.esp32.transport.ble_transport import BleTransport
from armdroid.hardware.esp32.transport.factory import make_transport
from armdroid.hardware.esp32.transport.serial_transport import SerialTransport
from armdroid.hardware.esp32.transport.tcp_transport import TcpTransport


def _make_cfg(protocol: str = "serial") -> MagicMock:
    cfg = MagicMock()
    cfg.transport.protocol = protocol
    cfg.transport.auth = None
    cfg.transport.tcp = MagicMock()
    cfg.transport.tcp.host = "127.0.0.1"
    cfg.transport.tcp.port = 3001
    cfg.transport.tcp.connect_timeout_s = 5.0
    cfg.transport.ble = MagicMock()
    cfg.transport.ble.device_address = "auto"
    cfg.transport.ble.device_name = "ArmDroid"
    cfg.transport.ble.rx_char_uuid = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
    cfg.transport.ble.tx_char_uuid = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
    cfg.transport.ble.scan_timeout_s = 10.0
    return cfg


def test_make_transport_serial() -> None:
    cfg = _make_cfg("serial")
    fake_serial = MagicMock()
    fake_serial.Serial = MagicMock()
    fake_list_ports = MagicMock()
    t = make_transport(cfg, serial_module=fake_serial, list_ports_module=fake_list_ports)
    assert isinstance(t, SerialTransport)


def test_make_transport_tcp() -> None:
    cfg = _make_cfg("tcp")
    t = make_transport(cfg)
    assert isinstance(t, TcpTransport)


def test_make_transport_ble() -> None:
    cfg = _make_cfg("ble")
    t = make_transport(cfg)
    assert isinstance(t, BleTransport)


def test_make_transport_unknown_protocol_raises() -> None:
    cfg = _make_cfg("udp")
    with pytest.raises(ValueError, match="udp"):
        make_transport(cfg)


def test_make_transport_wraps_tcp_with_auth() -> None:
    from armdroid.config.schema import TransportAuthConfig

    cfg = _make_cfg("tcp")
    cfg.transport.auth = TransportAuthConfig(key_hex="aa" * 16, required=True)
    t = make_transport(cfg)
    assert isinstance(t, AuthTransport)
    assert isinstance(t._inner, TcpTransport)


def test_make_transport_wraps_ble_with_auth() -> None:
    from armdroid.config.schema import TransportAuthConfig

    cfg = _make_cfg("ble")
    cfg.transport.auth = TransportAuthConfig(key_hex="bb" * 16, required=True)
    t = make_transport(cfg)
    assert isinstance(t, AuthTransport)
    assert isinstance(t._inner, BleTransport)


def test_make_transport_serial_no_auth_by_default() -> None:
    """Serial transport is NOT wrapped in AuthTransport when auth is None."""
    cfg = _make_cfg("serial")
    cfg.transport.auth = None
    fake_serial = MagicMock()
    t = make_transport(cfg, serial_module=fake_serial)
    assert isinstance(t, SerialTransport)
    assert not isinstance(t, AuthTransport)


def test_make_transport_ble_with_factory() -> None:
    cfg = _make_cfg("ble")
    factory = lambda addr: MagicMock()  # noqa: E731
    t = make_transport(cfg, ble_client_factory=factory)
    assert isinstance(t, BleTransport)
