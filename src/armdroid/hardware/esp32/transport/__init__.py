"""ESP32 transport package — byte-stream abstractions for the JSON driver.

Public surface
--------------
.. code-block:: python

    from armdroid.hardware.esp32.transport import ArmTransport, make_transport

    # Serial (default)
    transport = make_transport(cfg, serial_module=serial, list_ports_module=list_ports)

    # TCP
    transport = make_transport(cfg)  # cfg.transport.protocol = "tcp"

    # BLE
    transport = make_transport(cfg)  # cfg.transport.protocol = "ble"

    # BLE with injected fake client (tests)
    transport = make_transport(cfg, ble_client_factory=lambda addr: FakeBleClient(addr))
"""

from armdroid.hardware.esp32.transport.auth import AuthTransport, HmacFramer
from armdroid.hardware.esp32.transport.base import ArmTransport
from armdroid.hardware.esp32.transport.ble_transport import (
    BleakClientFactory,
    BleakClientProtocol,
    BleTransport,
)
from armdroid.hardware.esp32.transport.factory import make_transport
from armdroid.hardware.esp32.transport.serial_transport import SerialTransport
from armdroid.hardware.esp32.transport.tcp_transport import TcpTransport

__all__ = [
    "ArmTransport",
    "AuthTransport",
    "BleTransport",
    "BleakClientFactory",
    "BleakClientProtocol",
    "HmacFramer",
    "SerialTransport",
    "TcpTransport",
    "make_transport",
]
