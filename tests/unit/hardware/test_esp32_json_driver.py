"""Tests for :class:`Esp32JsonDriver` using an in-process fake firmware.

The fake (``_FakeSerial``) implements the read/write surface of
``serial.Serial`` and runs a tiny state machine that mirrors what the
real ESP32 firmware does:

* parses incoming JSON lines
* validates shape, joint count, range, e-stop latch
* enqueues ack / nak / state / evt frames for the host to read

This lets us exercise the full driver pipeline (encode -> write -> reader
demux -> future resolution) without any hardware. Tests can also inject
malformed lines and verify the host tolerates them.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import threading
import time
from collections import deque
from typing import Any, Final

import pytest
import pytest_asyncio

from armdroid.config.schema import (
    ArmConfig,
    ArmServoConfig,
    ArmTransportConfig,
    JointLimits,
)
from armdroid.protocols import (
    ArmCommandRejected,
    ArmDriverError,
    ArmDriverProtocol,
)

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
            serial_port="/dev/null",
            serial_baud=115_200,
            connect_timeout_s=1.0,
            command_timeout_s=0.5,
            heartbeat_hz=10.0,
            drain_pings_on_connect=1,
        ),
    }
    base.update(overrides)
    return ArmConfig.model_validate(base)


# --------------------------------------------------------------------- #
# Fake firmware — speaks PROTOCOL.md at the byte level.
# --------------------------------------------------------------------- #


class _FakeSerial:
    """Minimal stand-in for ``serial.Serial`` driving an in-process fw."""

    def __init__(
        self,
        *,
        port: str,
        baudrate: int,
        timeout: float,
        write_timeout: float,
    ) -> None:
        self._rx: deque[bytes] = deque()
        self._lock = threading.Lock()
        self._estop = False
        self._joints = [0.0] * 6
        self._velocities = [0.0] * 6
        self._is_moving = False
        self._seq = 0
        self._rx_timeout = timeout
        self._tx_buffer = bytearray()
        self.inject_raw_lines: list[bytes] = []
        self._enqueue({"t": "evt", "kind": "boot", "ver": "fake-1.0"})

    def readline(self) -> bytes:
        deadline = time.monotonic() + self._rx_timeout
        while True:
            with self._lock:
                if self.inject_raw_lines:
                    return self.inject_raw_lines.pop(0)
                if self._rx:
                    return self._rx.popleft()
            if time.monotonic() >= deadline:
                return b""
            time.sleep(0.005)

    def write(self, data: bytes) -> None:
        with self._lock:
            self._tx_buffer.extend(data)
            while b"\n" in self._tx_buffer:
                line, _, rest = self._tx_buffer.partition(b"\n")
                self._tx_buffer = bytearray(rest)
                self._handle_host_line(bytes(line))

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _enqueue(self, msg: dict[str, Any]) -> None:
        self._rx.append((json.dumps(msg) + "\n").encode("ascii"))

    def emit_state_now(self) -> None:
        with self._lock:
            self._seq += 1
            self._enqueue(
                {
                    "t": "state",
                    "seq": self._seq,
                    "ts": time.monotonic(),
                    "q": list(self._joints),
                    "qd": list(self._velocities),
                    "mv": self._is_moving,
                    "es": self._estop,
                }
            )

    def _handle_host_line(self, line: bytes) -> None:
        try:
            text = line.decode("ascii").strip()
            if not text:
                return
            msg = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if msg.get("t") != "cmd":
            return
        req_id = msg.get("id")
        cmd = msg.get("cmd")

        def nak(err: str, human: str = "") -> None:
            self._enqueue({"t": "nak", "id": req_id, "err": err, "msg": human})

        def ack() -> None:
            self._enqueue({"t": "ack", "id": req_id})

        if cmd == "ping":
            ack()
        elif cmd == "get_state":
            ack()
            self._seq += 1
            self._enqueue(
                {
                    "t": "state",
                    "seq": self._seq,
                    "ts": time.monotonic(),
                    "q": list(self._joints),
                    "qd": list(self._velocities),
                    "mv": self._is_moving,
                    "es": self._estop,
                }
            )
        elif cmd == "estop":
            self._estop = True
            self._is_moving = False
            ack()
        elif cmd == "clear_estop":
            self._estop = False
            ack()
        elif cmd == "set_joints":
            if self._estop:
                nak("estop_latched", "cannot move during e-stop")
                return
            q = msg.get("q")
            if not isinstance(q, list):
                nak("bad_shape", "q missing or not list")
                return
            if len(q) != 6:
                nak("bad_joint_count", f"got {len(q)}")
                return
            for v in q:
                if not isinstance(v, int | float):
                    nak("bad_shape", "non-numeric joint")
                    return
                if not math.isfinite(v):
                    nak("out_of_range", "non-finite joint")
                    return
                if abs(v) > math.pi + 1e-6:
                    nak("out_of_range", "joint exceeds firmware limit")
                    return
            self._joints = [float(v) for v in q]
            self._is_moving = True
            ack()
        else:
            nak("unknown_cmd", f"cmd={cmd}")


@pytest.fixture
def fake_serial_module(monkeypatch: pytest.MonkeyPatch) -> type[_FakeSerial]:
    """Install ``_FakeSerial`` as ``armdroid.hardware.esp32_json_driver._serial_module.Serial``."""
    from armdroid.hardware import esp32_json_driver

    fake_module = type(sys)("serial")
    fake_module.Serial = _FakeSerial  # type: ignore[attr-defined]
    monkeypatch.setattr(esp32_json_driver, "_serial_module", fake_module)
    return _FakeSerial


@pytest_asyncio.fixture
async def driver(fake_serial_module: type[_FakeSerial]) -> Any:
    from armdroid.hardware.esp32_json_driver import Esp32JsonDriver

    drv = Esp32JsonDriver(_make_config())
    await drv.connect()
    try:
        yield drv
    finally:
        await drv.disconnect()


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_satisfies_protocol(fake_serial_module: type[_FakeSerial]) -> None:
    from armdroid.hardware.esp32_json_driver import Esp32JsonDriver

    drv = Esp32JsonDriver(_make_config())
    assert isinstance(drv, ArmDriverProtocol)


@pytest.mark.asyncio
async def test_connect_drains_and_pings(
    fake_serial_module: type[_FakeSerial],
) -> None:
    from armdroid.hardware.esp32_json_driver import Esp32JsonDriver

    drv = Esp32JsonDriver(_make_config())
    await drv.connect()
    assert drv.is_connected
    await drv.disconnect()
    assert not drv.is_connected


@pytest.mark.asyncio
async def test_send_joint_positions_round_trip(driver: Any) -> None:
    target = (0.5, 0.0, 0.0, 0.0, 0.0, 0.0)
    await driver.send_joint_positions(target, duration_s=1.0)
    state = await driver.read_state()
    assert state.joint_positions == target


@pytest.mark.asyncio
async def test_local_validation_rejects_before_wire(driver: Any) -> None:
    bad = (0.0, math.nan, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ArmCommandRejected, match="non-finite"):
        await driver.send_joint_positions(bad, duration_s=1.0)


@pytest.mark.asyncio
async def test_firmware_nak_surfaces_as_command_rejected(
    fake_serial_module: type[_FakeSerial],
) -> None:
    """If host limits are wider than firmware, fw nak surfaces as ArmCommandRejected."""
    cfg = _make_config(
        joint_limits=[JointLimits(min_rad=-10.0, max_rad=10.0, max_velocity_rad_s=20.0)] * 6,
    )
    from armdroid.hardware.esp32_json_driver import Esp32JsonDriver

    drv = Esp32JsonDriver(cfg)
    await drv.connect()
    try:
        bad = (5.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # past fake-fw ±π limit
        with pytest.raises(ArmCommandRejected, match="out_of_range"):
            await drv.send_joint_positions(bad, duration_s=1.0)
    finally:
        await drv.disconnect()


@pytest.mark.asyncio
async def test_estop_blocks_motion_then_clear_resumes(driver: Any) -> None:
    await driver.emergency_stop()
    await asyncio.sleep(0.05)
    with pytest.raises(ArmCommandRejected, match="estop_latched"):
        await driver.send_joint_positions((0.0,) * 6, duration_s=1.0)
    await driver.clear_emergency_stop()
    await driver.send_joint_positions((0.1,) * 6, duration_s=1.0)


@pytest.mark.asyncio
async def test_state_frames_update_cache(
    fake_serial_module: type[_FakeSerial],
) -> None:
    from armdroid.hardware.esp32_json_driver import Esp32JsonDriver

    drv = Esp32JsonDriver(_make_config())
    await drv.connect()
    try:
        fake = drv._port  # type: ignore[attr-defined]
        assert isinstance(fake, _FakeSerial)
        fake.emit_state_now()
        await asyncio.sleep(0.05)
        state = await drv.read_state()
        assert len(state.joint_positions) == 6
        assert not state.estop_active
    finally:
        await drv.disconnect()


@pytest.mark.asyncio
async def test_malformed_lines_are_dropped(
    fake_serial_module: type[_FakeSerial],
) -> None:
    from armdroid.hardware.esp32_json_driver import Esp32JsonDriver

    drv = Esp32JsonDriver(_make_config())
    await drv.connect()
    try:
        fake = drv._port  # type: ignore[attr-defined]
        assert isinstance(fake, _FakeSerial)
        with fake._lock:
            fake.inject_raw_lines.append(b"this is not json\n")
            fake.inject_raw_lines.append(b"\xff\xfe\x00\n")
        await asyncio.sleep(0.05)
        # Driver should still work after garbage
        await drv.send_joint_positions((0.0,) * 6, duration_s=1.0)
    finally:
        await drv.disconnect()


@pytest.mark.asyncio
async def test_command_timeout_raises_driver_error(
    fake_serial_module: type[_FakeSerial],
) -> None:
    from armdroid.hardware.esp32_json_driver import Esp32JsonDriver

    cfg = _make_config(
        transport=ArmTransportConfig(
            protocol="serial",
            serial_port="/dev/null",
            serial_baud=115_200,
            connect_timeout_s=1.0,
            command_timeout_s=0.1,
            heartbeat_hz=10.0,
            drain_pings_on_connect=1,
        ),
    )
    drv = Esp32JsonDriver(cfg)
    await drv.connect()
    try:
        fake = drv._port  # type: ignore[attr-defined]
        original = fake._handle_host_line

        def silent_for_set_joints(line: bytes) -> None:
            if b'"set_joints"' in line:
                return
            original(line)

        fake._handle_host_line = silent_for_set_joints  # type: ignore[method-assign]
        with pytest.raises(ArmDriverError, match="No reply"):
            await drv.send_joint_positions((0.0,) * 6, duration_s=1.0)
    finally:
        await drv.disconnect()


@pytest.mark.asyncio
async def test_disconnect_cancels_pending(
    fake_serial_module: type[_FakeSerial],
) -> None:
    from armdroid.hardware.esp32_json_driver import Esp32JsonDriver

    cfg = _make_config(
        transport=ArmTransportConfig(
            protocol="serial",
            serial_port="/dev/null",
            serial_baud=115_200,
            connect_timeout_s=1.0,
            command_timeout_s=5.0,
            heartbeat_hz=10.0,
            drain_pings_on_connect=1,
        ),
    )
    drv = Esp32JsonDriver(cfg)
    await drv.connect()
    fake = drv._port  # type: ignore[attr-defined]
    original = fake._handle_host_line
    fake._handle_host_line = lambda line: None  # type: ignore[method-assign]

    async def issue() -> None:
        await drv.send_joint_positions((0.0,) * 6, duration_s=1.0)

    task = asyncio.create_task(issue())
    await asyncio.sleep(0.05)
    fake._handle_host_line = original  # type: ignore[method-assign]
    await drv.disconnect()
    with pytest.raises(ArmDriverError):
        await task


@pytest.mark.asyncio
async def test_legacy_send_joint_command_silently_clips(
    fake_serial_module: type[_FakeSerial],
) -> None:
    """Legacy adapter clips out-of-range angles silently per joint."""
    cfg = _make_config()
    cfg.joint_limits[0] = JointLimits(min_rad=-0.5, max_rad=0.5, max_velocity_rad_s=10.0)
    from armdroid.hardware.esp32_json_driver import Esp32JsonDriver

    drv = Esp32JsonDriver(cfg)
    await drv.connect()
    try:
        import numpy as np

        await drv.send_joint_command(np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        await asyncio.sleep(0.05)
        joints = await drv.get_joint_states()
        assert joints[0] == pytest.approx(0.5, abs=1e-6)
    finally:
        await drv.disconnect()


@pytest.mark.asyncio
async def test_legacy_lifecycle_aliases(
    fake_serial_module: type[_FakeSerial],
) -> None:
    from armdroid.hardware.esp32_json_driver import Esp32JsonDriver

    drv = Esp32JsonDriver(_make_config())
    await drv.start()
    assert drv.is_connected
    await drv.stop()
    assert not drv.is_connected


@pytest.mark.asyncio
async def test_velocity_limit_rejection(driver: Any) -> None:
    """1 rad in 0.05 s = 20 rad/s, above the 10 rad/s generous limit."""
    with pytest.raises(ArmCommandRejected, match="rad/s"):
        await driver.send_joint_positions((1.0,) * 6, duration_s=0.05)


@pytest.mark.asyncio
async def test_estop_requires_connected(
    fake_serial_module: type[_FakeSerial],
) -> None:
    """emergency_stop on a disconnected driver raises ArmDriverError."""
    from armdroid.hardware.esp32_json_driver import Esp32JsonDriver

    drv = Esp32JsonDriver(_make_config())
    with pytest.raises(ArmDriverError):
        await drv.emergency_stop()


@pytest.mark.asyncio
async def test_estop_serialised_with_send(
    fake_serial_module: type[_FakeSerial],
) -> None:
    """emergency_stop issued while a send_joint_positions is in flight
    goes through the send lock — both complete without raising.
    """
    from armdroid.hardware.esp32_json_driver import Esp32JsonDriver

    cfg = _make_config(
        transport=ArmTransportConfig(
            protocol="serial",
            serial_port="/dev/null",
            serial_baud=115_200,
            connect_timeout_s=1.0,
            command_timeout_s=1.0,
            heartbeat_hz=10.0,
            drain_pings_on_connect=1,
        ),
    )
    drv = Esp32JsonDriver(cfg)
    await drv.connect()
    try:
        send_task = asyncio.create_task(drv.send_joint_positions((0.1,) * 6, duration_s=1.0))
        estop_task = asyncio.create_task(drv.emergency_stop())
        await asyncio.gather(send_task, estop_task, return_exceptions=True)
        # After e-stop the arm rejects motion
        with pytest.raises(ArmCommandRejected):
            await drv.send_joint_positions((0.0,) * 6, duration_s=1.0)
    finally:
        await drv.disconnect()


@pytest.mark.asyncio
async def test_keepalive_pings_when_idle(
    fake_serial_module: type[_FakeSerial],
) -> None:
    """With a very short keepalive_interval_s the driver pings the fake-fw."""
    from armdroid.hardware.esp32_json_driver import Esp32JsonDriver

    cfg = _make_config(
        transport=ArmTransportConfig(
            protocol="serial",
            serial_port="/dev/null",
            serial_baud=115_200,
            connect_timeout_s=1.0,
            command_timeout_s=0.5,
            heartbeat_hz=10.0,
            drain_pings_on_connect=1,
            keepalive_interval_s=0.05,
        ),
    )
    drv = Esp32JsonDriver(cfg)
    await drv.connect()
    try:
        ping_count_before = drv._next_id  # type: ignore[attr-defined]
        # Wait 3x the keepalive interval — at least one ping should fire.
        await asyncio.sleep(0.18)
        ping_count_after = drv._next_id  # type: ignore[attr-defined]
        assert ping_count_after > ping_count_before, (
            "Expected at least one keepalive ping to be sent"
        )
    finally:
        await drv.disconnect()


@pytest.mark.asyncio
async def test_encode_returns_tuple_with_req_id(
    fake_serial_module: type[_FakeSerial],
) -> None:
    """_encode returns (wire_line, req_id) and increments _next_id."""
    from armdroid.hardware.esp32_json_driver import Esp32JsonDriver

    drv = Esp32JsonDriver(_make_config())
    start_id = drv._next_id  # type: ignore[attr-defined]
    line, req_id = drv._encode("ping", {})  # type: ignore[attr-defined]
    assert req_id == start_id
    assert drv._next_id == start_id + 1  # type: ignore[attr-defined]
    assert line.endswith("\n")
    parsed = json.loads(line.strip())
    assert parsed["id"] == req_id
    assert parsed["cmd"] == "ping"


@pytest.mark.asyncio
async def test_write_blocking_disconnected_raises(
    fake_serial_module: type[_FakeSerial],
) -> None:
    """_write_blocking on a disconnected driver raises ArmDriverError."""
    from armdroid.hardware.esp32_json_driver import Esp32JsonDriver

    drv = Esp32JsonDriver(_make_config())
    # _port is None when not connected
    with pytest.raises(ArmDriverError, match="not connected"):
        drv._write_blocking(b"test\n")  # type: ignore[attr-defined]
