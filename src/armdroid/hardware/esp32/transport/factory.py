"""Transport factory for the ESP32 JSON driver.

:func:`make_transport` selects the correct transport implementation from
``cfg.transport.protocol`` and optionally wraps it with
:class:`~.auth.AuthTransport` when auth is configured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from armdroid.hardware.esp32.transport.auth import AuthTransport, HmacFramer
from armdroid.hardware.esp32.transport.base import ArmTransport
from armdroid.hardware.esp32.transport.ble_transport import (
    BleakClientFactory,
    BleTransport,
)
from armdroid.hardware.esp32.transport.serial_transport import SerialTransport
from armdroid.hardware.esp32.transport.tcp_transport import TcpTransport

if TYPE_CHECKING:
    from armdroid.config.schema import ArmConfig


def make_transport(
    cfg: ArmConfig,
    *,
    serial_module: Any = None,
    list_ports_module: Any = None,
    ble_client_factory: BleakClientFactory | None = None,
) -> ArmTransport:
    """Construct and return the appropriate :class:`ArmTransport` for *cfg*.

    If ``cfg.transport.auth`` is set, the returned transport is wrapped in an
    :class:`~.auth.AuthTransport` that signs outgoing frames and verifies
    incoming ones.

    Args:
        cfg: Arm configuration.
        serial_module: The ``serial`` module (or monkeypatched fake) to pass
            to :class:`SerialTransport`.  Required when ``protocol='serial'``.
        list_ports_module: The ``serial.tools.list_ports`` module (or fake).
        ble_client_factory: Optional BLE client factory for test injection.

    Returns:
        A ready-to-use :class:`ArmTransport` instance (not yet connected).

    Raises:
        ValueError: If ``cfg.transport.protocol`` is unrecognised.
        ArmDriverError: If a required dependency is missing.
    """
    protocol = cfg.transport.protocol

    transport: ArmTransport
    if protocol == "serial":
        transport = SerialTransport(cfg, serial_module, list_ports_module)
    elif protocol == "tcp":
        transport = TcpTransport(cfg)
    elif protocol == "ble":
        transport = BleTransport(cfg, client_factory=ble_client_factory)
    else:
        msg = f"Unknown transport protocol: {protocol!r}"
        raise ValueError(msg)

    if cfg.transport.auth is not None:
        framer = HmacFramer.from_config(cfg.transport.auth)
        transport = AuthTransport(transport, framer)

    return transport


__all__ = ["make_transport"]
