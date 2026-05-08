"""Tests for ``Esp32JsonDriver`` auto-port discovery.

Cover ``cfg.arm.transport.serial_port == "auto"``:

* No candidates found -> ArmDriverError with helpful message.
* All candidates time out -> ArmDriverError mentioning probe count.
* One candidate responds with ack -> driver binds to that port.
* Excluded ports are skipped during enumeration.
* VID:PID hint filters narrow the candidate set.
* explicit serial_port path bypasses autodetect entirely.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Final

import pytest

from armdroid.config.schema import (
    ArmConfig,
    ArmServoConfig,
    ArmTransportConfig,
    JointLimits,
)
from armdroid.domain.errors import ArmDriverError
from tests.helpers.fake_serial import PingOnlyFakeSerial as _RespondingFakeSerial
from tests.helpers.fake_serial import SilentFakeSerial as _SilentFakeSerial

_GENEROUS_LIMITS: Final = JointLimits(
    min_rad=-math.pi,
    max_rad=math.pi,
    max_velocity_rad_s=10.0,
)


def _make_config(**overrides: object) -> ArmConfig:
    base: dict[str, object] = {
        "dof": 6,
        "joint_limits": [_GENEROUS_LIMITS] * 6,
        "home_position": [0.0] * 6,
        "servos": [ArmServoConfig(pwm_pin=13 + i) for i in range(6)],
        "transport": ArmTransportConfig(
            protocol="serial",
            serial_port="auto",
            serial_baud=115_200,
            connect_timeout_s=0.5,
            command_timeout_s=0.2,
            heartbeat_hz=10.0,
            drain_pings_on_connect=1,
        ),
    }
    base.update(overrides)
    return ArmConfig.model_validate(base)


class _FakePortInfo:
    """Stand-in for ``serial.tools.list_ports.ListPortInfo``."""

    def __init__(self, device: str, hwid: str = "") -> None:
        self.device = device
        self.hwid = hwid


def _install_fake_serial(
    monkeypatch: pytest.MonkeyPatch,
    serial_class: type[Any],
    ports: list[_FakePortInfo],
) -> None:
    """Replace _serial_module.Serial and _list_ports_module.comports."""
    from armdroid.hardware.esp32 import driver as _esp32_driver_mod

    fake_serial = type(sys)("serial")
    fake_serial.Serial = serial_class  # type: ignore[attr-defined]
    monkeypatch.setattr(_esp32_driver_mod, "_serial_module", fake_serial)
    fake_list_ports = type(sys)("serial.tools.list_ports")
    fake_list_ports.comports = lambda: list(ports)  # type: ignore[attr-defined]
    monkeypatch.setattr(_esp32_driver_mod, "_list_ports_module", fake_list_ports)


@pytest.mark.asyncio
async def test_no_candidates_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_serial(monkeypatch, _RespondingFakeSerial, ports=[])
    from armdroid.hardware.esp32 import Esp32JsonDriver

    drv = Esp32JsonDriver(_make_config())
    with pytest.raises(ArmDriverError, match="no candidate"):
        await drv.connect()


@pytest.mark.asyncio
async def test_all_candidates_silent_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_serial(
        monkeypatch,
        _SilentFakeSerial,
        ports=[
            _FakePortInfo("/dev/ttyUSB0"),
            _FakePortInfo("/dev/ttyUSB1"),
        ],
    )
    from armdroid.hardware.esp32 import Esp32JsonDriver

    drv = Esp32JsonDriver(_make_config())
    with pytest.raises(ArmDriverError, match="probed 2"):
        await drv.connect()


@pytest.mark.asyncio
async def test_responding_port_is_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_serial(
        monkeypatch,
        _RespondingFakeSerial,
        ports=[_FakePortInfo("/dev/ttyUSB0")],
    )
    from armdroid.hardware.esp32 import Esp32JsonDriver

    drv = Esp32JsonDriver(_make_config())
    await drv.connect()
    try:
        assert drv.is_connected
    finally:
        await drv.disconnect()


# ---------------------------------------------------------------------------
# Portfinder gap tests — lines 78-79 (JSON decode error) and 146-152 (probe
# raises OSError → debug-log + return None path inside ``_probe``).
# ---------------------------------------------------------------------------


class _GarbageThenAckSerial:
    """Fake serial that first emits one line of garbage, then ACKs the ping.

    Exercises the ``except (UnicodeDecodeError, json.JSONDecodeError): continue``
    branch in ``probe_port_blocking`` (portfinder.py lines 78-79).
    """

    def __init__(self, *, port: str, baudrate: int, timeout: float, write_timeout: float) -> None:
        import threading
        from collections import deque

        self._rx: deque[bytes] = deque()
        self._lock = threading.Lock()
        self._timeout = timeout
        # Prime with a garbage non-JSON line before any ping arrives.
        self._rx.append(b"not-json!!!\n")

    def readline(self) -> bytes:
        import time

        deadline = time.monotonic() + self._timeout
        while True:
            with self._lock:
                if self._rx:
                    return self._rx.popleft()
            if time.monotonic() >= deadline:
                return b""
            time.sleep(0.005)

    def write(self, data: bytes) -> None:
        import json

        with self._lock:
            try:
                msg = json.loads(data.rstrip(b"\n").decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return
            if msg.get("t") == "cmd" and msg.get("cmd") == "ping":
                self._rx.append(
                    (json.dumps({"t": "ack", "id": msg.get("id")}) + "\n").encode("ascii")
                )

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_probe_tolerates_garbage_before_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """portfinder.py lines 78-79: JSON decode error in probe is skipped; the
    ack that follows is still recognised and the device is returned."""
    _install_fake_serial(monkeypatch, _GarbageThenAckSerial, ports=[_FakePortInfo("/dev/ttyUSB9")])
    from armdroid.hardware.esp32 import Esp32JsonDriver

    drv = Esp32JsonDriver(_make_config())
    await drv.connect()
    try:
        assert drv.is_connected
    finally:
        await drv.disconnect()


@pytest.mark.asyncio
async def test_autodetect_probe_os_error_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """portfinder.py lines 146-152: when probe_port_blocking raises OSError the
    exception is caught by the inner _probe coroutine, logged, and None is
    returned — all-probes-failed path raises ArmDriverError."""
    from unittest.mock import patch

    _install_fake_serial(
        monkeypatch,
        _RespondingFakeSerial,
        ports=[_FakePortInfo("/dev/ttyUSB0"), _FakePortInfo("/dev/ttyUSB1")],
    )
    from armdroid.hardware.esp32 import Esp32JsonDriver

    with patch(
        "armdroid.hardware.esp32.portfinder.probe_port_blocking",
        side_effect=OSError("permission denied"),
    ):
        drv = Esp32JsonDriver(_make_config())
        with pytest.raises(ArmDriverError, match="probed 2"):
            await drv.connect()


@pytest.mark.asyncio
async def test_excluded_ports_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the only ports are excluded, autodetect raises."""
    _install_fake_serial(
        monkeypatch,
        _RespondingFakeSerial,
        ports=[_FakePortInfo("/dev/ttyUSB0")],
    )
    cfg = _make_config(
        transport=ArmTransportConfig(
            protocol="serial",
            serial_port="auto",
            serial_baud=115_200,
            connect_timeout_s=0.5,
            command_timeout_s=0.2,
            heartbeat_hz=10.0,
            drain_pings_on_connect=1,
            exclude_ports=["/dev/ttyUSB0"],
        ),
    )
    from armdroid.hardware.esp32 import Esp32JsonDriver

    drv = Esp32JsonDriver(cfg)
    with pytest.raises(ArmDriverError, match="no candidate"):
        await drv.connect()


@pytest.mark.asyncio
async def test_vid_pid_hints_filter_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hint matching is substring on hwid; non-matching ports are dropped."""
    _install_fake_serial(
        monkeypatch,
        _RespondingFakeSerial,
        ports=[
            _FakePortInfo("/dev/ttyUSB0", hwid="USB VID:PID=1234:5678 SER=ABC"),
            _FakePortInfo("/dev/ttyUSB1", hwid="USB VID:PID=DEAD:BEEF SER=XYZ"),
        ],
    )
    cfg = _make_config(
        transport=ArmTransportConfig(
            protocol="serial",
            serial_port="auto",
            serial_baud=115_200,
            connect_timeout_s=0.5,
            command_timeout_s=0.2,
            heartbeat_hz=10.0,
            drain_pings_on_connect=1,
            usb_vid_pid_hints=["1234:5678"],
        ),
    )
    from armdroid.hardware.esp32 import Esp32JsonDriver

    drv = Esp32JsonDriver(cfg)
    await drv.connect()
    try:
        assert drv.is_connected
    finally:
        await drv.disconnect()


@pytest.mark.asyncio
async def test_explicit_path_bypasses_autodetect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When serial_port is an explicit path, autodetect is not invoked."""
    _install_fake_serial(
        monkeypatch,
        _RespondingFakeSerial,
        ports=[],  # no candidates — but autodetect should not be called
    )
    cfg = _make_config(
        transport=ArmTransportConfig(
            protocol="serial",
            serial_port="/dev/ttyUSB42",
            serial_baud=115_200,
            connect_timeout_s=0.5,
            command_timeout_s=0.2,
            heartbeat_hz=10.0,
            drain_pings_on_connect=1,
        ),
    )
    from armdroid.hardware.esp32 import Esp32JsonDriver

    drv = Esp32JsonDriver(cfg)
    await drv.connect()
    try:
        assert drv.is_connected
    finally:
        await drv.disconnect()
