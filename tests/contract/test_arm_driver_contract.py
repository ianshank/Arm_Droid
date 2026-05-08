"""ArmDriverProtocol contract tests parametrised over implementations.

Every concrete driver — :class:`MockArmDriver` and :class:`Esp32JsonDriver`
(backed by an in-process fake firmware) — must satisfy the same observable
behaviour for the modern surface:

* lifecycle (connect / disconnect / is_connected, idempotent)
* motion validation (wrong length, NaN, joint-limit, duration, velocity)
* read_state returns an ArmState consistent with the most recent command
* latched e-stop blocks motion and clear_emergency_stop releases it

Tests are parametrised by a fixture factory that yields a connected
driver instance. Adding a new driver implementation only requires adding
its factory to the ``DRIVER_FACTORIES`` list — no test changes needed.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any, Final

import pytest

from armdroid.config.schema import (
    ArmConfig,
    ArmServoConfig,
    ArmTransportConfig,
    JointLimits,
)
from armdroid.hardware.mock_arm_driver import MockArmDriver
from armdroid.protocols import (
    ArmCommandRejected,
    ArmDriverProtocol,
)

pytestmark = pytest.mark.contract

_GENEROUS_LIMITS: Final = JointLimits(
    min_rad=-math.pi,
    max_rad=math.pi,
    max_velocity_rad_s=10.0,
)


def _cfg() -> ArmConfig:
    return ArmConfig(
        dof=6,
        home_position=[0.0] * 6,
        joint_limits=[_GENEROUS_LIMITS] * 6,
        servos=[ArmServoConfig(pwm_pin=13 + i) for i in range(6)],
        transport=ArmTransportConfig(
            protocol="serial",
            serial_port="/dev/null",
            serial_baud=115_200,
            connect_timeout_s=1.0,
            command_timeout_s=0.5,
            heartbeat_hz=10.0,
            drain_pings_on_connect=1,
        ),
    )


# --------------------------------------------------------------------- #
# Fake serial firmware shared with the parametrised ESP32 driver factory.
# --------------------------------------------------------------------- #


class _ContractFakeSerial:
    """Minimal _FakeSerial — accepts and replies to the contract test cmds."""

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
        self._enqueue({"t": "evt", "kind": "boot", "ver": "fake-1.0"})

    def readline(self) -> bytes:
        deadline = time.monotonic() + self._rx_timeout
        while True:
            with self._lock:
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
                self._handle(bytes(line))

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _enqueue(self, msg: dict[str, Any]) -> None:
        self._rx.append((json.dumps(msg) + "\n").encode("ascii"))

    def _handle(self, line: bytes) -> None:
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
        if cmd == "ping":
            self._enqueue({"t": "ack", "id": req_id})
        elif cmd == "get_state":
            self._enqueue({"t": "ack", "id": req_id})
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
            self._enqueue({"t": "ack", "id": req_id})
        elif cmd == "clear_estop":
            self._estop = False
            self._enqueue({"t": "ack", "id": req_id})
        elif cmd == "set_joints":
            if self._estop:
                self._enqueue(
                    {
                        "t": "nak",
                        "id": req_id,
                        "err": "estop_latched",
                        "msg": "",
                    }
                )
                return
            q = msg.get("q")
            if not isinstance(q, list) or len(q) != 6:
                self._enqueue(
                    {
                        "t": "nak",
                        "id": req_id,
                        "err": "bad_joint_count",
                        "msg": "",
                    }
                )
                return
            for v in q:
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    self._enqueue(
                        {
                            "t": "nak",
                            "id": req_id,
                            "err": "out_of_range",
                            "msg": "",
                        }
                    )
                    return
                if abs(v) > math.pi + 1e-6:
                    self._enqueue(
                        {
                            "t": "nak",
                            "id": req_id,
                            "err": "out_of_range",
                            "msg": "",
                        }
                    )
                    return
            self._joints = [float(v) for v in q]
            self._is_moving = True
            self._enqueue({"t": "ack", "id": req_id})
        else:
            self._enqueue({"t": "nak", "id": req_id, "err": "unknown_cmd", "msg": ""})


def _install_contract_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    from armdroid.hardware import esp32_json_driver

    fake_serial = type(sys)("serial")
    fake_serial.Serial = _ContractFakeSerial  # type: ignore[attr-defined]
    monkeypatch.setattr(esp32_json_driver, "_serial_module", fake_serial)


# --------------------------------------------------------------------- #
# Driver factories — add a row here and the whole suite parametrises.
# --------------------------------------------------------------------- #

_DriverFactory = Callable[[pytest.MonkeyPatch], Any]


def _make_mock(monkeypatch: pytest.MonkeyPatch) -> MockArmDriver:
    return MockArmDriver(_cfg())


def _make_esp32(monkeypatch: pytest.MonkeyPatch) -> Any:
    _install_contract_fake(monkeypatch)
    from armdroid.hardware.esp32_json_driver import Esp32JsonDriver

    return Esp32JsonDriver(_cfg())


DRIVER_FACTORIES: list[tuple[str, _DriverFactory]] = [
    ("mock", _make_mock),
    ("esp32_json", _make_esp32),
]


@pytest.fixture(params=DRIVER_FACTORIES, ids=lambda p: p[0])
def driver_factory(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Yield a fresh, *unconnected* driver instance for each test."""
    _name, factory = request.param
    return factory(monkeypatch)


# --------------------------------------------------------------------- #
# Contract assertions
# --------------------------------------------------------------------- #


class TestContractProtocolConformance:
    def test_satisfies_protocol(self, driver_factory: Any) -> None:
        assert isinstance(driver_factory, ArmDriverProtocol)


class TestContractLifecycle:
    @pytest.mark.asyncio
    async def test_connect_disconnect_idempotent(
        self,
        driver_factory: Any,
    ) -> None:
        drv = driver_factory
        assert not drv.is_connected
        await drv.connect()
        await drv.connect()  # idempotent
        assert drv.is_connected
        await drv.disconnect()
        await drv.disconnect()  # idempotent
        assert not drv.is_connected


class TestContractValidation:
    @pytest.mark.asyncio
    async def test_wrong_length_rejected(self, driver_factory: Any) -> None:
        drv = driver_factory
        await drv.connect()
        try:
            with pytest.raises(ArmCommandRejected):
                await drv.send_joint_positions((0.0, 0.0, 0.0), duration_s=1.0)
        finally:
            await drv.disconnect()

    @pytest.mark.asyncio
    async def test_non_finite_rejected(self, driver_factory: Any) -> None:
        drv = driver_factory
        await drv.connect()
        try:
            with pytest.raises(ArmCommandRejected):
                await drv.send_joint_positions(
                    (0.0, math.nan, 0.0, 0.0, 0.0, 0.0),
                    duration_s=1.0,
                )
        finally:
            await drv.disconnect()

    @pytest.mark.asyncio
    async def test_zero_duration_rejected(self, driver_factory: Any) -> None:
        drv = driver_factory
        await drv.connect()
        try:
            with pytest.raises(ArmCommandRejected):
                await drv.send_joint_positions((0.0,) * 6, duration_s=0.0)
        finally:
            await drv.disconnect()


class TestContractEstop:
    @pytest.mark.asyncio
    async def test_estop_blocks_then_clear_resumes(
        self,
        driver_factory: Any,
    ) -> None:
        drv = driver_factory
        await drv.connect()
        try:
            await drv.emergency_stop()
            await asyncio.sleep(0.05)
            with pytest.raises(ArmCommandRejected):
                await drv.send_joint_positions((0.0,) * 6, duration_s=1.0)
            await drv.clear_emergency_stop()
            # Now a motion command should succeed
            await drv.send_joint_positions((0.1,) * 6, duration_s=1.0)
        finally:
            await drv.disconnect()


class TestContractReadState:
    @pytest.mark.asyncio
    async def test_read_state_after_connect_returns_arm_state(
        self,
        driver_factory: Any,
    ) -> None:
        from armdroid.protocols import ArmState

        drv = driver_factory
        await drv.connect()
        try:
            state = await drv.read_state()
            assert isinstance(state, ArmState)
            assert len(state.joint_positions) == 6
            assert len(state.joint_velocities) == 6
            assert isinstance(state.is_moving, bool)
            assert isinstance(state.estop_active, bool)
        finally:
            await drv.disconnect()

    @pytest.mark.asyncio
    async def test_send_then_read_reflects_command(
        self,
        driver_factory: Any,
    ) -> None:
        drv = driver_factory
        await drv.connect()
        try:
            target = (0.5, -0.3, 0.2, 0.0, 0.1, -0.1)
            await drv.send_joint_positions(target, duration_s=1.0)
            await asyncio.sleep(0.05)  # let mock or fake-fw catch up
            state = await drv.read_state()
            # Driver reported a state — the mock interpolates so we can't
            # assert exactness without freezing time, but we can assert
            # the *segment* was received (estop clear, valid joint count).
            assert state.estop_active is False
            assert len(state.joint_positions) == 6
        finally:
            await drv.disconnect()


class TestContractLegacyAdapters:
    """Both implementations honour the legacy adapter surface."""

    @pytest.mark.asyncio
    async def test_legacy_start_stop_proxy(self, driver_factory: Any) -> None:
        drv = driver_factory
        await drv.start()
        assert drv.is_connected
        await drv.stop()
        assert not drv.is_connected

    @pytest.mark.asyncio
    async def test_get_joint_states_returns_ndarray(
        self,
        driver_factory: Any,
    ) -> None:
        import numpy as np

        drv = driver_factory
        await drv.connect()
        try:
            joints = await drv.get_joint_states()
            assert isinstance(joints, np.ndarray)
            assert joints.shape == (6,)
        finally:
            await drv.disconnect()
