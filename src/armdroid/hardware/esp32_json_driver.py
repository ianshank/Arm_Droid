"""ESP32-JSON arm driver — newline-delimited JSON over UART.

Speaks the wire protocol described in ``firmware/arm_esp32/PROTOCOL.md``.
Implements the full :class:`armdroid.protocols.ArmDriverProtocol` surface
(both modern and legacy adapters) so the controller, primitives, and
existing tests don't care which transport is wired up.

Architecture:

* ``pyserial`` is synchronous, so all read/write calls run via
  :func:`asyncio.to_thread` to avoid blocking the orchestrator's event loop.
* A long-lived background task (:meth:`_reader_loop`) parses every line
  the firmware emits and demuxes them:

  - ``state`` heartbeats -> cached in ``self._latest_state``
  - ``ack`` / ``nak`` replies -> routed to the matching pending future
    keyed by request ``id``
  - ``evt`` events -> logged

* :meth:`send_joint_positions` and the e-stop methods serialise a JSON
  line, register a future, write the line, and await the reply with a
  per-command timeout (default 250 ms — must fit in a 30 Hz tick).
* Local validation runs eagerly *before* writing to the wire so we fail
  fast and don't burn ack budget on commands the firmware would reject.

Concurrency: an :class:`asyncio.Lock` serialises sends so two overlapping
calls never interleave bytes on the wire. :meth:`read_state` is lock-free
and returns the cached heartbeat. The e-stop write bypasses the send
lock so safety-critical commands always go through.

Port discovery: when ``cfg.arm.transport.serial_port == "auto"`` the
driver enumerates available USB serial ports via
:mod:`serial.tools.list_ports`, optionally filtered by VID:PID hints and
exclusion list, and probes each in parallel for the firmware boot
signature. The first port that responds with a valid ``ack`` to a
``ping`` becomes the bound port.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from armdroid.logging.setup import get_logger
from armdroid.protocols import (
    ArmCommandRejected,
    ArmDriverError,
    ArmState,
)

try:
    import serial as _serial_module
except ImportError:  # pragma: no cover - hardware extra
    _serial_module = None

try:
    from serial.tools import list_ports as _list_ports_module
except ImportError:  # pragma: no cover - hardware extra
    _list_ports_module = None

if TYPE_CHECKING:
    from armdroid.config.schema import ArmConfig

_log = get_logger(__name__)


@dataclass(slots=True)
class _PendingReply:
    """A command awaiting its ack/nak from the firmware."""

    future: asyncio.Future[None]
    cmd_name: str


class Esp32JsonDriver:
    """``ArmDriverProtocol`` implementation over pyserial + JSON-over-UART.

    Args:
        cfg: Validated :class:`ArmConfig` with ``transport`` populated.
    """

    def __init__(self, cfg: ArmConfig) -> None:
        """Initialise the driver. Does not open the port — call ``connect``."""
        if _serial_module is None:
            msg = (
                "pyserial not installed. Install with `pip install -e .[hardware]`"
                " to enable the real-hardware driver."
            )
            raise ArmDriverError(msg)
        self._cfg = cfg
        self._dof = cfg.dof
        self._port: Any | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()

        # Pending replies keyed by request id.
        self._pending: dict[int, _PendingReply] = {}
        self._next_id = 1

        # Cached heartbeat. None until the first state frame arrives.
        self._latest_state: ArmState | None = None
        self._connected = False
        self._gripper_open = True
        _log.info(
            "esp32_json_driver_init",
            dof=self._dof,
            port=cfg.transport.serial_port,
            baud=cfg.transport.serial_baud,
        )

    # ------------------------------------------------------------------ #
    # Modern lifecycle
    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        """Open the transport. Idempotent."""
        if self._connected:
            return
        port_path = await self._resolve_port()
        port = await asyncio.to_thread(self._open_port_blocking, port_path)
        self._port = port
        self._reader_task = asyncio.create_task(self._reader_loop(), name="esp32_json_reader")
        self._connected = True
        try:
            for _ in range(self._cfg.transport.drain_pings_on_connect):
                await self._send_and_await_ack(cmd="ping", payload={})
        except Exception:
            await self._teardown()
            raise

        _log.info(
            "esp32_json_driver_connected",
            port=port_path,
            baud=self._cfg.transport.serial_baud,
        )

    async def disconnect(self) -> None:
        """Close the transport. Idempotent."""
        if not self._connected:
            return
        await self._teardown()
        _log.info("esp32_json_driver_disconnected")

    @property
    def is_connected(self) -> bool:
        """Whether the transport is currently open."""
        return self._connected

    # ------------------------------------------------------------------ #
    # Modern motion
    # ------------------------------------------------------------------ #

    async def send_joint_positions(
        self,
        positions: tuple[float, ...],
        duration_s: float,
    ) -> None:
        """Validate locally, encode, write, await ack."""
        self._require_connected()
        self._validate_command(positions, duration_s)
        await self._send_and_await_ack(
            cmd="set_joints",
            payload={
                "q": list(positions),
                "dur_ms": round(duration_s * 1000.0),
            },
        )

    async def read_state(self) -> ArmState:
        """Return the latest cached state heartbeat.

        On first call (before any heartbeat has landed) we force a
        ``get_state`` and wait up to ``cfg.transport.first_state_wait_s``
        for the reply to be cached.
        """
        self._require_connected()
        if self._latest_state is None:
            await self._send_and_await_ack(cmd="get_state", payload={})
            wait_s = self._cfg.transport.first_state_wait_s
            poll_interval = 0.01
            steps = max(1, int(wait_s / poll_interval))
            for _ in range(steps):
                if self._latest_state is not None:
                    break
                await asyncio.sleep(poll_interval)
            if self._latest_state is None:
                msg = "No state frame received from firmware after get_state"
                raise ArmDriverError(msg)
        return self._latest_state

    # ------------------------------------------------------------------ #
    # Modern safety
    # ------------------------------------------------------------------ #

    async def emergency_stop(self) -> None:
        """Latch e-stop unconditionally.

        Bypasses the send lock so a contested driver can still issue the
        safety command on the wire.
        """
        line = self._encode("estop", {}).encode("ascii")
        try:
            await asyncio.to_thread(self._write_blocking, line)
        except Exception as exc:  # pragma: no cover - hardware fault path
            _log.error("esp32_json_estop_write_failed", error=str(exc))
            msg = f"E-stop write failed: {exc}"
            raise ArmDriverError(msg) from exc
        _log.warning("esp32_json_estop_sent")

    async def clear_emergency_stop(self) -> None:
        """Release the e-stop latch."""
        await self._send_and_await_ack(cmd="clear_estop", payload={})

    # ------------------------------------------------------------------ #
    # Legacy adapters — preserved for backwards compatibility.
    # ------------------------------------------------------------------ #

    @property
    def dof(self) -> int:
        """Joint vector length."""
        return self._dof

    async def start(self) -> None:
        """Legacy lifecycle alias."""
        await self.connect()

    async def stop(self) -> None:
        """Legacy lifecycle alias."""
        await self.disconnect()

    async def get_joint_states(self) -> NDArray[np.float64]:
        """Legacy state read — returns numpy array of joint positions."""
        if not self._connected:
            await self.connect()
        state = await self.read_state()
        return np.array(state.joint_positions, dtype=np.float64)

    async def send_joint_command(self, target_angles: NDArray[np.float64]) -> None:
        """Legacy step command — silent per-joint clipping, configured duration."""
        if not self._connected:
            await self.connect()
        if len(target_angles) != self._dof:
            msg = f"Expected {self._dof} joint angles, got {len(target_angles)}"
            raise ValueError(msg)
        clipped = tuple(
            max(
                self._cfg.joint_limits[i].min_rad,
                min(self._cfg.joint_limits[i].max_rad, float(v)),
            )
            for i, v in enumerate(target_angles)
        )
        try:
            await self.send_joint_positions(clipped, duration_s=self._cfg.home_duration_s)
        except ArmCommandRejected as exc:
            msg = str(exc)
            raise ValueError(msg) from exc

    async def close_gripper(self) -> float:
        """Legacy gripper close. Returns simulated grip force on a 6-DoF arm.

        On a 7-DoF arm with the gripper at joint index ``dof - 1``, commit
        7 will route this through ``send_joint_positions`` to write the
        gripper joint to ``1.0``. For 6-DoF hardware (no protocol gripper
        joint) this is a bookkeeping flag.
        """
        if not self._connected:
            await self.connect()
        if self._gripper_open:
            self._gripper_open = False
            grip_force = 1.0
        else:
            grip_force = 0.0
        return grip_force

    async def open_gripper(self) -> None:
        """Legacy gripper open."""
        if not self._connected:
            await self.connect()
        self._gripper_open = True

    async def home(self) -> None:
        """Move arm to home position."""
        if not self._connected:
            await self.connect()
        home = tuple(float(v) for v in self._cfg.home_position)
        await self.send_joint_positions(home, duration_s=self._cfg.home_duration_s)
        self._gripper_open = True

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _require_connected(self) -> None:
        if not self._connected:
            msg = "Esp32JsonDriver is not connected"
            raise ArmDriverError(msg)

    def _validate_command(
        self,
        positions: tuple[float, ...],
        duration_s: float,
    ) -> None:
        if len(positions) != self._dof:
            msg = f"Expected {self._dof} joint positions, got {len(positions)}"
            raise ArmCommandRejected(msg)
        if duration_s <= 0.0:
            msg = f"duration_s must be positive, got {duration_s}"
            raise ArmCommandRejected(msg)
        for idx, value in enumerate(positions):
            if not math.isfinite(value):
                msg = f"joint[{idx}] is non-finite: {value}"
                raise ArmCommandRejected(msg)
            limits = self._cfg.joint_limits[idx]
            if not (limits.min_rad <= value <= limits.max_rad):
                msg = f"joint[{idx}]={value} outside [{limits.min_rad}, {limits.max_rad}]"
                raise ArmCommandRejected(msg)

    async def _resolve_port(self) -> str:
        """Resolve the configured serial port — explicit path or 'auto'."""
        configured = self._cfg.transport.serial_port
        if configured != "auto":
            return configured
        return await self._autodetect_port()

    async def _autodetect_port(self) -> str:
        """Probe USB serial ports for the firmware boot signature.

        Sends a ``ping`` to each candidate in parallel and binds the first
        one that returns a valid ``ack`` within ``connect_timeout_s``.
        """
        if _list_ports_module is None:  # pragma: no cover - hardware extra
            msg = (
                "serial.tools.list_ports not available — install pyserial >= 3.5"
                " to enable port autodetect."
            )
            raise ArmDriverError(msg)
        excluded = set(self._cfg.transport.exclude_ports)
        hints = [h.lower() for h in self._cfg.transport.usb_vid_pid_hints]
        candidates: list[str] = []
        for port_info in _list_ports_module.comports():
            device = str(port_info.device)
            if device in excluded:
                continue
            if hints:
                hwid = (port_info.hwid or "").lower()
                if not any(h in hwid for h in hints):
                    continue
            candidates.append(device)
        if not candidates:
            msg = (
                "Port autodetect found no candidate USB serial ports. "
                "Plug the ESP32 in or set cfg.arm.transport.serial_port "
                "to an explicit path."
            )
            raise ArmDriverError(msg)
        _log.info("esp32_json_autodetect_candidates", candidates=candidates)
        # Probe in parallel, bounded by autodetect_probe_concurrency.
        sem = asyncio.Semaphore(self._cfg.transport.autodetect_probe_concurrency)

        async def _probe(device: str) -> str | None:
            async with sem:
                try:
                    return await asyncio.wait_for(
                        asyncio.to_thread(self._probe_port_blocking, device),
                        timeout=self._cfg.transport.connect_timeout_s,
                    )
                except (TimeoutError, ArmDriverError, OSError) as exc:
                    _log.debug(
                        "esp32_json_autodetect_probe_failed",
                        port=device,
                        error=str(exc),
                    )
                    return None

        results = await asyncio.gather(*(_probe(d) for d in candidates))
        for result in results:
            if result is not None:
                _log.info("esp32_json_autodetect_bound", port=result)
                return result
        msg = (
            f"Port autodetect probed {len(candidates)} candidate(s) but none "
            "responded with a valid firmware ack. Set serial_port explicitly "
            "or check that the firmware is flashed."
        )
        raise ArmDriverError(msg)

    def _probe_port_blocking(self, device: str) -> str | None:
        """Open ``device``, send a ping, wait for a matching ack.

        Returns ``device`` on success and ``None`` if the port doesn't
        respond like the firmware. Always closes the port — caller will
        re-open in :meth:`connect`.
        """
        assert _serial_module is not None
        port = None
        try:
            port = _serial_module.Serial(
                port=device,
                baudrate=self._cfg.transport.serial_baud,
                timeout=self._cfg.transport.connect_timeout_s,
                write_timeout=self._cfg.transport.command_timeout_s,
            )
            ping_msg = (
                json.dumps(
                    {"t": "cmd", "id": 1, "ts": time.monotonic(), "cmd": "ping"},
                    separators=(",", ":"),
                )
                + "\n"
            )
            port.write(ping_msg.encode("ascii"))
            port.flush()
            deadline = time.monotonic() + self._cfg.transport.connect_timeout_s
            while time.monotonic() < deadline:
                line = port.readline()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("ascii", errors="replace").strip())
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if msg.get("t") == "ack" and msg.get("id") == 1:
                    return device
                if msg.get("t") == "evt" and msg.get("kind") == "boot":
                    # firmware just booted; re-issue the ping after a beat
                    port.write(ping_msg.encode("ascii"))
                    port.flush()
            return None
        finally:
            if port is not None:
                with contextlib.suppress(Exception):  # pragma: no cover
                    port.close()

    def _open_port_blocking(self, device: str) -> Any:
        assert _serial_module is not None
        return _serial_module.Serial(
            port=device,
            baudrate=self._cfg.transport.serial_baud,
            timeout=self._cfg.transport.connect_timeout_s,
            write_timeout=self._cfg.transport.command_timeout_s,
        )

    async def _teardown(self) -> None:
        self._connected = False
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader_task
            self._reader_task = None
        if self._port is not None:
            with contextlib.suppress(Exception):  # pragma: no cover
                await asyncio.to_thread(self._port.close)
            self._port = None
        for pending in list(self._pending.values()):
            if not pending.future.done():
                pending.future.set_exception(ArmDriverError("Driver disconnected before reply"))
        self._pending.clear()

    def _encode(self, cmd: str, payload: dict[str, Any]) -> str:
        msg: dict[str, Any] = {
            "t": "cmd",
            "id": self._next_id,
            "ts": time.monotonic(),
            "cmd": cmd,
            **payload,
        }
        self._next_id += 1
        line = json.dumps(msg, separators=(",", ":")) + "\n"
        max_bytes = self._cfg.transport.max_line_bytes
        if len(line.encode("ascii")) > max_bytes:
            err = f"Encoded command exceeds {max_bytes} bytes"
            raise ArmCommandRejected(err)
        return line

    async def _send_and_await_ack(
        self,
        cmd: str,
        payload: dict[str, Any],
    ) -> None:
        async with self._send_lock:
            line = self._encode(cmd, payload)
            req_id = json.loads(line)["id"]
            loop = asyncio.get_running_loop()
            future: asyncio.Future[None] = loop.create_future()
            self._pending[req_id] = _PendingReply(future=future, cmd_name=cmd)
            try:
                await asyncio.to_thread(self._write_blocking, line.encode("ascii"))
            except Exception as exc:
                self._pending.pop(req_id, None)
                err = f"Serial write failed: {exc}"
                raise ArmDriverError(err) from exc

            try:
                await asyncio.wait_for(
                    future,
                    timeout=self._cfg.transport.command_timeout_s,
                )
            except TimeoutError as exc:
                self._pending.pop(req_id, None)
                err = (
                    f"No reply from firmware for cmd={cmd!r} within "
                    f"{self._cfg.transport.command_timeout_s}s"
                )
                raise ArmDriverError(err) from exc
            finally:
                self._pending.pop(req_id, None)

    def _write_blocking(self, data: bytes) -> None:
        assert self._port is not None
        self._port.write(data)
        self._port.flush()

    async def _reader_loop(self) -> None:
        """Read lines forever, dispatch by message type."""
        try:
            while self._connected and self._port is not None:
                line = await asyncio.to_thread(self._readline_blocking)
                if not line:
                    continue
                self._dispatch_line(line)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - hardware fault path
            _log.error("esp32_json_reader_crashed", error=str(exc))

    def _readline_blocking(self) -> bytes:
        assert self._port is not None
        result: bytes = self._port.readline()
        return result

    def _dispatch_line(self, raw: bytes) -> None:
        max_bytes = self._cfg.transport.max_line_bytes
        if len(raw) > max_bytes:
            _log.warning("esp32_json_dropped_oversized_line", n=len(raw))
            return
        try:
            text = raw.decode("ascii", errors="replace").strip()
            if not text:
                return
            msg = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _log.warning("esp32_json_bad_frame", error=str(exc), raw=raw[:80])
            return

        msg_type = msg.get("t")
        if msg_type == "state":
            self._handle_state(msg)
        elif msg_type == "ack":
            self._resolve_pending(msg, success=True)
        elif msg_type == "nak":
            self._resolve_pending(msg, success=False)
        elif msg_type == "evt":
            _log.info(
                "esp32_json_event",
                kind=msg.get("kind"),
                fields={k: v for k, v in msg.items() if k not in {"t", "kind"}},
            )
        else:
            _log.warning("esp32_json_unknown_frame", t=msg_type)

    def _handle_state(self, msg: dict[str, Any]) -> None:
        try:
            q = tuple(float(x) for x in msg["q"])
            qd = tuple(float(x) for x in msg["qd"])
            if len(q) != self._dof or len(qd) != self._dof:
                err = "wrong joint count"
                raise ValueError(err)
            self._latest_state = ArmState(
                joint_positions=q,
                joint_velocities=qd,
                is_moving=bool(msg.get("mv", False)),
                estop_active=bool(msg.get("es", False)),
                timestamp_s=float(msg.get("ts", time.monotonic())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            _log.warning("esp32_json_bad_state_frame", error=str(exc))

    def _resolve_pending(self, msg: dict[str, Any], *, success: bool) -> None:
        req_id = msg.get("id")
        if not isinstance(req_id, int):
            _log.warning("esp32_json_reply_missing_id", msg=msg)
            return
        pending = self._pending.get(req_id)
        if pending is None:
            _log.debug("esp32_json_orphan_reply", id=req_id)
            return
        if pending.future.done():
            return
        if success:
            pending.future.set_result(None)
        else:
            err = msg.get("err", "unknown")
            human = msg.get("msg", "")
            if err in {"out_of_range", "bad_joint_count", "estop_latched"}:
                pending.future.set_exception(
                    ArmCommandRejected(f"firmware rejected: {err}: {human}")
                )
            else:
                pending.future.set_exception(ArmDriverError(f"firmware error {err}: {human}"))
