"""ESP32 JSON driver — ArmDriverProtocol implementation over pyserial + JSON-over-UART.

Speaks the wire protocol described in ``firmware/arm_esp32/PROTOCOL.md``.
Implements the full :class:`armdroid.protocols.ArmDriverProtocol` surface
(both modern and legacy adapters) so the controller, primitives, and
existing tests don't care which transport is wired up.

Architecture:

* ``pyserial`` is synchronous, so all read/write calls run via
  :func:`asyncio.to_thread` to avoid blocking the orchestrator's event loop.
* A long-lived background task (:meth:`_reader_loop`) parses every line
  the firmware emits and demuxes them:

  - ``state`` heartbeats → cached in ``self._latest_state``
  - ``ack`` / ``nak`` replies → routed to the matching pending future
    keyed by request ``id``
  - ``evt`` events → logged

* :meth:`send_joint_positions` and the e-stop methods serialise a JSON
  line, register a future, write the line, and await the reply with a
  per-command timeout (default 250 ms).
* Local validation runs eagerly *before* writing to the wire so we fail
  fast and don't burn ack budget on commands the firmware would reject.

Pure helper modules:

* :mod:`.portfinder` — serial port discovery and probing
* :mod:`.framing` — wire-frame decoding and pending-reply bookkeeping
* :mod:`.validator` — local command validation and velocity-anchor logic
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from numpy.typing import NDArray

from armdroid.domain.errors import (
    ArmCommandRejected,
    ArmDriverError,
)
from armdroid.domain.state import ArmState
from armdroid.hardware.esp32.framing import _PendingReply, decode_frame
from armdroid.hardware.esp32.transport import ArmTransport, make_transport
from armdroid.hardware.esp32.validator import validate_joint_positions, velocity_anchor
from armdroid.logging.setup import get_logger
from armdroid.telemetry import (
    SPAN_DRIVER_CONNECT,
    SPAN_DRIVER_DISCONNECT,
    SPAN_DRIVER_SEND,
    get_telemetry,
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

# Polling floor for the keepalive background task. Not user-tunable — it
# is an asyncio scheduling implementation detail, not a hardware parameter.
_KEEPALIVE_POLL_FLOOR_S: Final[float] = 0.05

# Resolution of the "wait for first state frame" polling loop inside
# read_state(). Sub-tick so the driver is responsive even on slow hosts.
_FIRST_STATE_POLL_INTERVAL_S: Final[float] = 0.01


class Esp32JsonDriver:
    """``ArmDriverProtocol`` implementation over pyserial + JSON-over-UART.

    Speaks the wire protocol described in
    ``firmware/arm_esp32/PROTOCOL.md``. Implements both the modern
    :class:`~armdroid.protocols.ArmDriverProtocol` surface and the
    legacy adapter API for backwards compatibility.

    Pure helper logic is extracted into sibling modules:

    * :mod:`.portfinder` — serial port discovery and probing
    * :mod:`.framing` — wire-frame decoding
    * :mod:`.validator` — command validation and velocity-anchor selection
    """

    def __init__(self, cfg: ArmConfig) -> None:
        """Initialise the driver. Does not open the port — call ``connect``."""
        self._cfg = cfg
        self._dof = cfg.dof
        self._transport: ArmTransport = make_transport(
            cfg,
            serial_module=_serial_module,
            list_ports_module=_list_ports_module,
        )
        self._reader_task: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()

        # Pending replies keyed by request id.
        self._pending: dict[int, _PendingReply] = {}
        self._next_id = 1

        # Cached heartbeat. None until the first state frame arrives.
        self._latest_state: ArmState | None = None
        self._connected = False
        self._gripper_open = True

        # Last target we commanded (post-ack). Used to anchor velocity
        # checks before the first heartbeat lands; None until the first
        # successful send_joint_positions completes.
        self._last_commanded_target: tuple[float, ...] | None = None

        # Monotonic timestamp of the last successful wire write. Kept
        # current by _send_and_await_ack (after the write) and by
        # emergency_stop. Drives the keepalive loop.
        self._last_send_monotonic: float = 0.0
        self._keepalive_task: asyncio.Task[None] | None = None

        _log.info(
            "esp32_json_driver_init",
            dof=self._dof,
            transport=cfg.transport.protocol,
            baud=cfg.transport.serial_baud,
        )

    # ------------------------------------------------------------------ #
    # Modern lifecycle
    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        """Open the transport. Idempotent."""
        if self._connected:
            return
        with get_telemetry().start_span(
            SPAN_DRIVER_CONNECT,
            transport=self._cfg.transport.protocol,
            baud=self._cfg.transport.serial_baud,
        ):
            await self._transport.connect()
            self._last_send_monotonic = time.monotonic()
            self._reader_task = asyncio.create_task(self._reader_loop(), name="esp32_json_reader")
            self._keepalive_task = asyncio.create_task(
                self._keepalive_loop(), name="esp32_json_keepalive"
            )
            self._connected = True
            try:
                for _ in range(self._cfg.transport.drain_pings_on_connect):
                    await self._send_and_await_ack(cmd="ping", payload={})
            except Exception:
                await self._teardown()
                raise

            _log.info(
                "esp32_json_driver_connected",
                transport=self._cfg.transport.protocol,
                baud=self._cfg.transport.serial_baud,
            )

    async def disconnect(self) -> None:
        """Close the transport. Idempotent."""
        if not self._connected:
            return
        with get_telemetry().start_span(SPAN_DRIVER_DISCONNECT):
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
        with get_telemetry().start_span(
            SPAN_DRIVER_SEND,
            dof=self._dof,
            duration_s=duration_s,
        ):
            self._validate_command(positions, duration_s)
            await self._send_and_await_ack(
                cmd="set_joints",
                payload={
                    "q": list(positions),
                    "dur_ms": round(duration_s * 1000.0),
                },
            )
        # Update after a successful ack so the velocity check has a
        # reliable anchor for the next command.
        self._last_commanded_target = positions

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
            poll_interval = _FIRST_STATE_POLL_INTERVAL_S
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
        """Latch e-stop — acquires the send lock and writes immediately.

        Holding the send lock ensures the transport is not called concurrently
        with another sender. Raises :exc:`ArmDriverError` if the driver
        is not connected rather than propagating an AssertionError.
        """
        if not self._connected:
            msg = "Esp32JsonDriver is not connected — cannot issue e-stop"
            raise ArmDriverError(msg)
        async with self._send_lock:
            line, _req_id = self._encode("estop", {})
            try:
                await self._transport.write_line(line.encode("ascii"))
                self._last_send_monotonic = time.monotonic()
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
        """Legacy step command — silent per-joint clipping, configured duration.

        Matches :class:`MockArmDriver.send_joint_command` semantics: out-of-range
        angles are silently clipped and velocity limits are *not* enforced.
        The configured ``home_duration_s`` is used as a target, but is
        automatically stretched if any joint would exceed its
        ``max_velocity_rad_s`` — this preserves the legacy "no velocity check"
        contract while still passing the strict validation in
        :meth:`send_joint_positions`.
        """
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
        # Legacy semantics: "as fast as one firmware interpolator tick".
        interp_hz = self._cfg.firmware.interpolator_hz
        base_duration = 1.0 / interp_hz if interp_hz > 0.0 else 0.02
        anchor = self._velocity_anchor()
        min_duration = base_duration
        for idx, (s, t) in enumerate(zip(anchor, clipped, strict=True)):
            limit = self._cfg.joint_limits[idx].max_velocity_rad_s
            if limit > 0.0:
                # +1% headroom to absorb float rounding in the validator.
                required = abs(t - s) / limit * 1.01
                if required > min_duration:
                    min_duration = required
        try:
            await self.send_joint_positions(clipped, duration_s=min_duration)
        except ArmCommandRejected as exc:
            msg = str(exc)
            raise ValueError(msg) from exc

    async def close_gripper(self) -> float:
        """Legacy gripper close. Returns simulated grip force on a 6-DoF arm."""
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

    def _velocity_anchor(self) -> tuple[float, ...]:
        """Return the best available start-position for velocity checks."""
        return velocity_anchor(
            self._latest_state,
            self._last_commanded_target,
            list(self._cfg.home_position),
        )

    def _validate_command(
        self,
        positions: tuple[float, ...],
        duration_s: float,
    ) -> None:
        validate_joint_positions(
            positions=positions,
            duration_s=duration_s,
            dof=self._dof,
            joint_limits=list(self._cfg.joint_limits),
            anchor=self._velocity_anchor(),
        )

    def _encode(self, cmd: str, payload: dict[str, Any]) -> tuple[str, int]:
        """Serialise a command and return ``(wire_line, req_id)``.

        Returning the ``req_id`` avoids the caller having to re-parse the
        serialised JSON just to recover the id.
        """
        req_id = self._next_id
        msg: dict[str, Any] = {
            "t": "cmd",
            "id": req_id,
            "ts": time.monotonic(),
            "cmd": cmd,
            **payload,
        }
        self._next_id += 1
        line = json.dumps(msg, separators=(",", ":")) + "\n"
        max_bytes = self._cfg.transport.max_line_bytes
        # Exclude the trailing '\n' delimiter: firmware counts content bytes only.
        if len(line.encode("ascii")) - 1 > max_bytes:
            err = f"Encoded command exceeds {max_bytes} bytes"
            raise ArmCommandRejected(err)
        return line, req_id

    async def _send_and_await_ack(
        self,
        cmd: str,
        payload: dict[str, Any],
    ) -> None:
        async with self._send_lock:
            line, req_id = self._encode(cmd, payload)
            loop = asyncio.get_running_loop()
            future: asyncio.Future[None] = loop.create_future()
            self._pending[req_id] = _PendingReply(future=future, cmd_name=cmd)
            try:
                await self._transport.write_line(line.encode("ascii"))
                self._last_send_monotonic = time.monotonic()
            except Exception as exc:
                self._pending.pop(req_id, None)
                err = f"Transport write failed: {exc}"
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

    async def _keepalive_loop(self) -> None:
        """Ping the firmware periodically when the wire is otherwise idle.

        Fires when ``time.monotonic() - _last_send_monotonic`` exceeds
        ``cfg.transport.keepalive_interval_s``. Using the send lock via
        ``_send_and_await_ack`` ensures the ping never races a concurrent
        command. The loop exits cleanly on ``CancelledError`` (issued by
        ``_teardown``).
        """
        interval = self._cfg.transport.keepalive_interval_s
        # Poll at ¼ of the interval so we never slip by a full period.
        poll_s = max(_KEEPALIVE_POLL_FLOOR_S, interval / 4.0)
        try:
            while True:
                await asyncio.sleep(poll_s)
                if not self._connected:
                    break
                idle_s = time.monotonic() - self._last_send_monotonic
                if idle_s >= interval:
                    _log.debug(
                        "esp32_json_keepalive_ping",
                        idle_s=round(idle_s, 3),
                    )
                    try:
                        await self._send_and_await_ack(cmd="ping", payload={})
                    except ArmDriverError as exc:
                        _log.warning(
                            "esp32_json_keepalive_failed",
                            error=str(exc),
                        )
        except asyncio.CancelledError:
            raise

    async def _reader_loop(self) -> None:
        """Read lines forever, dispatch by message type.

        On TCP / BLE EOF the underlying transport returns ``b""`` and
        flips ``transport.is_connected`` to False. Without an explicit
        check this would spin at 100% CPU on every empty read; break
        when the transport reports it is disconnected so the driver's
        :meth:`disconnect` (or a supervisor task) can rebuild the
        transport rather than the loop pegging a core. Empty reads on a
        still-connected transport (legitimate keepalive idle) keep the
        original ``continue`` semantics so existing behaviour is
        preserved on the serial path.
        """
        try:
            while self._connected:
                line = await self._transport.readline()
                if not line:
                    if not self._transport.is_connected:
                        _log.warning(
                            "esp32_json_reader_transport_disconnected",
                            transport=type(self._transport).__name__,
                        )
                        break
                    continue
                self._dispatch_line(line)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - hardware fault path
            _log.error("esp32_json_reader_crashed", error=str(exc))

    def _dispatch_line(self, raw: bytes) -> None:
        max_bytes = self._cfg.transport.max_line_bytes
        if len(raw) > max_bytes:
            _log.warning("esp32_json_dropped_oversized_line", n=len(raw))
            return
        if not raw.strip():
            return
        try:
            msg = decode_frame(raw)
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

    async def _teardown(self) -> None:
        self._connected = False
        for task_attr in ("_keepalive_task", "_reader_task"):
            task: asyncio.Task[None] | None = getattr(self, task_attr)
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                setattr(self, task_attr, None)
        with contextlib.suppress(Exception):
            await self._transport.disconnect()
        for pending in list(self._pending.values()):
            if not pending.future.done():
                pending.future.set_exception(ArmDriverError("Driver disconnected before reply"))
        self._pending.clear()


__all__ = ["Esp32JsonDriver"]
