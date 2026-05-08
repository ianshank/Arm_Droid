"""Canonical fake-serial implementations for armdroid tests.

These stand-ins replace ``serial.Serial`` in tests so the full driver
pipeline (encode → write → reader-loop → future-resolution) can be
exercised without physical hardware.

Three concrete classes are provided:

* :class:`FakeSerial` — speaks the full PROTOCOL.md state machine.  Used
  by the unit-driver tests and the contract-test suite.  Supports
  ``inject_raw_lines`` (raw byte injection for protocol-robustness tests)
  and ``emit_state_now()`` (force an unsolicited state frame).

* :class:`PingOnlyFakeSerial` — minimal stand-in that emits a boot event
  and replies to ``ping`` but ignores all other commands.  Enough for
  port-autodetect tests.

* :class:`SilentFakeSerial` — subclass of :class:`PingOnlyFakeSerial` that
  discards writes without responding, simulating a port that opens but
  never speaks the protocol (negative autodetect test).

All classes share the same constructor signature so they can be used as
drop-in replacements for each other in fixture code.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from typing import Any

# ---------------------------------------------------------------------------
# Polling floor for readline busy-wait — matches _FakeSerial history.
# ---------------------------------------------------------------------------
_READLINE_POLL_S: float = 0.005


class FakeSerial:
    """Full PROTOCOL.md stand-in for ``serial.Serial``.

    Constructor keyword arguments mirror ``serial.Serial`` so the module
    can be installed as a drop-in via ``monkeypatch``.

    Args:
        port: Port identifier (ignored — kept for API compatibility).
        baudrate: Baud rate (ignored — kept for API compatibility).
        timeout: Read timeout in seconds, applied per ``readline`` call.
        write_timeout: Write timeout (ignored — kept for API compatibility).
        dof: Degrees-of-freedom for the simulated arm (default 6).
    """

    def __init__(
        self,
        *,
        port: str,
        baudrate: int,
        timeout: float,
        write_timeout: float,
        dof: int = 6,
    ) -> None:
        self._rx: deque[bytes] = deque()
        self._lock = threading.Lock()
        self._estop = False
        self._dof = dof
        self._joints: list[float] = [0.0] * dof
        self._velocities: list[float] = [0.0] * dof
        self._is_moving = False
        self._seq = 0
        self._rx_timeout = timeout
        self._tx_buffer = bytearray()
        # Tests can prepend raw bytes to be returned by readline() before the
        # normal queue — used for protocol-robustness / malformed-line tests.
        self.inject_raw_lines: list[bytes] = []
        self._enqueue({"t": "evt", "kind": "boot", "ver": "fake-1.0"})

    # ------------------------------------------------------------------
    # serial.Serial public surface
    # ------------------------------------------------------------------

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
            time.sleep(_READLINE_POLL_S)

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

    # ------------------------------------------------------------------
    # Test-harness helpers
    # ------------------------------------------------------------------

    def emit_state_now(self) -> None:
        """Push an unsolicited state frame into the receive queue."""
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enqueue(self, msg: dict[str, Any]) -> None:
        self._rx.append((json.dumps(msg) + "\n").encode("ascii"))

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
            if len(q) != self._dof:
                nak("bad_joint_count", f"got {len(q)}")
                return
            for v in q:
                if not isinstance(v, (int, float)):
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


class PingOnlyFakeSerial:
    """Minimal fake — emits a boot event and replies to ``ping`` only.

    Used by port-autodetect tests where the full protocol state machine
    is not needed.  Constructor signature matches :class:`FakeSerial`.
    """

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
        self._tx_buffer = bytearray()
        self._timeout = timeout
        self._enqueue({"t": "evt", "kind": "boot", "ver": "fake-1.0"})

    def readline(self) -> bytes:
        deadline = time.monotonic() + self._timeout
        while True:
            with self._lock:
                if self._rx:
                    return self._rx.popleft()
            if time.monotonic() >= deadline:
                return b""
            time.sleep(_READLINE_POLL_S)

    def write(self, data: bytes) -> None:
        with self._lock:
            self._tx_buffer.extend(data)
            while b"\n" in self._tx_buffer:
                line, _, rest = self._tx_buffer.partition(b"\n")
                self._tx_buffer = bytearray(rest)
                try:
                    msg = json.loads(line.decode("ascii").strip())
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if msg.get("t") == "cmd" and msg.get("cmd") == "ping":
                    self._enqueue({"t": "ack", "id": msg.get("id")})

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _enqueue(self, msg: dict[str, Any]) -> None:
        self._rx.append((json.dumps(msg) + "\n").encode("ascii"))


class SilentFakeSerial(PingOnlyFakeSerial):
    """Port that opens but never responds — for negative autodetect tests."""

    def write(self, data: bytes) -> None:
        # Discard all writes; never enqueue a response.
        return None
