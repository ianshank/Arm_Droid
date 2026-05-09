r"""HMAC-SHA256 frame authentication for TCP and BLE transports.

:class:`HmacFramer` signs outgoing JSON frames and verifies incoming ones.
:class:`AuthTransport` wraps any :class:`~.base.ArmTransport` with signing /
verification so the driver and transport implementations stay auth-agnostic.

Wire format
-----------
Every authenticated frame is a standard JSON line with two extra fields
appended **before** serialisation:

* ``"seq"`` — a per-connection monotonically increasing integer.
* ``"sig"`` — lowercase hex HMAC-SHA256 of the canonical JSON (all fields
  including ``seq``, excluding ``sig`` itself; keys sorted, no whitespace).

Example signed frame::

    {"t":"cmd","id":1,"ts":1.2,"cmd":"ping","seq":1,"sig":"aabbcc..."}\\n

Security properties
-------------------
* Full HMAC-SHA256 (32 bytes / 64 hex chars) — no truncation.
* Constant-time comparison via :func:`hmac.compare_digest`.
* Monotonic sequence anti-replay: receiver rejects ``seq ≤ last_seen``.
* Serial transport bypasses auth (physical trust, no key exchange needed).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import TYPE_CHECKING, Any

from armdroid.domain.errors import ArmDriverError
from armdroid.logging.setup import get_logger

if TYPE_CHECKING:
    from armdroid.config.schema import TransportAuthConfig
    from armdroid.hardware.esp32.transport.base import ArmTransport

_log = get_logger(__name__)

# Minimum HMAC key length (bytes).  Shorter keys are rejected at construction.
_MIN_KEY_BYTES: int = 16


class HmacFramer:
    """Signs and verifies JSON frames with full HMAC-SHA256 + sequence anti-replay.

    Args:
        key: Raw HMAC key bytes.  Must be at least 16 bytes.

    Raises:
        ValueError: If the key is too short.
    """

    def __init__(self, key: bytes) -> None:
        if len(key) < _MIN_KEY_BYTES:
            msg = f"HMAC key must be at least {_MIN_KEY_BYTES} bytes (got {len(key)})"
            raise ValueError(msg)
        self._key = key
        self._out_seq: int = 0
        self._last_in_seq: int = -1

    # ------------------------------------------------------------------ #
    # Factory constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_hex(cls, hex_key: str) -> HmacFramer:
        """Construct from a hex-encoded key string.

        Args:
            hex_key: Even-length hex string (e.g. ``"deadbeef..."``).

        Raises:
            ValueError: If *hex_key* is malformed or too short.
        """
        try:
            key = bytes.fromhex(hex_key.strip())
        except ValueError as exc:
            msg = f"HMAC key is not valid hex: {exc}"
            raise ValueError(msg) from exc
        return cls(key)

    @classmethod
    def from_config(cls, auth_cfg: TransportAuthConfig) -> HmacFramer:
        """Construct from a :class:`~armdroid.config.schema.TransportAuthConfig`.

        Tries ``auth_cfg.key_hex`` first, then ``os.environ[auth_cfg.key_env_var]``.

        Raises:
            ArmDriverError: If no key is found and ``auth_cfg.required`` is True.
            ValueError: If the key is malformed.
        """
        hex_key: str | None = auth_cfg.key_hex
        if not hex_key:
            hex_key = os.environ.get(auth_cfg.key_env_var, "")
        if not hex_key:
            if auth_cfg.required:
                msg = (
                    f"HMAC key not set: neither auth.key_hex nor env var "
                    f"{auth_cfg.key_env_var!r} is populated.  "
                    "Set ARMDROID_HMAC_KEY or configure auth.key_hex."
                )
                raise ArmDriverError(msg)
            # required=False: return a dummy framer that signs but does not enforce.
            # Used only in test-only configs; never deployed.
            return cls(b"\x00" * _MIN_KEY_BYTES)
        return cls.from_hex(hex_key)

    # ------------------------------------------------------------------ #
    # Sign / verify
    # ------------------------------------------------------------------ #

    def sign(self, line: bytes) -> bytes:
        """Return *line* with ``seq`` and ``sig`` fields added.

        Args:
            line: A complete newline-terminated JSON line.

        Returns:
            Signed newline-terminated JSON line.

        Raises:
            ValueError: If *line* is not valid JSON.
        """
        try:
            frame: dict[str, Any] = json.loads(line.decode("ascii"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            msg = f"Cannot sign non-JSON frame: {exc}"
            raise ValueError(msg) from exc

        self._out_seq += 1
        frame["seq"] = self._out_seq
        # Canonical form: sorted keys, compact separators, no sig field yet.
        canonical = json.dumps(frame, sort_keys=True, separators=(",", ":")).encode("ascii")
        sig_hex = hmac.digest(self._key, canonical, hashlib.sha256).hex()
        frame["sig"] = sig_hex
        return (json.dumps(frame, separators=(",", ":")) + "\n").encode("ascii")

    def verify(self, line: bytes) -> bytes:
        """Verify *line* and return it with ``sig`` stripped.

        Args:
            line: A complete newline-terminated JSON line containing ``seq``
                and ``sig`` fields.

        Returns:
            Verified line with ``sig`` removed (``seq`` retained).

        Raises:
            ValueError: If verification fails for any reason (bad sig,
                replay, missing fields, bad encoding).
        """
        try:
            frame: dict[str, Any] = json.loads(line.decode("ascii").strip())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            msg = f"Cannot verify non-JSON frame: {exc}"
            raise ValueError(msg) from exc

        sig = frame.pop("sig", None)
        if not isinstance(sig, str):
            msg = "Frame missing 'sig' field"
            raise ValueError(msg)

        seq = frame.get("seq")
        if not isinstance(seq, int):
            msg = "Frame missing 'seq' field"
            raise ValueError(msg)

        if seq <= self._last_in_seq:
            msg = f"Anti-replay: seq {seq} ≤ last accepted {self._last_in_seq}"
            raise ValueError(msg)

        canonical = json.dumps(frame, sort_keys=True, separators=(",", ":")).encode("ascii")
        expected_hex = hmac.digest(self._key, canonical, hashlib.sha256).hex()
        if not hmac.compare_digest(sig, expected_hex):
            msg = "HMAC-SHA256 signature verification failed"
            raise ValueError(msg)

        self._last_in_seq = seq
        return (json.dumps(frame, separators=(",", ":")) + "\n").encode("ascii")


# ---------------------------------------------------------------------------
# AuthTransport — decorator over any ArmTransport
# ---------------------------------------------------------------------------


class AuthTransport:
    """Wraps any :class:`~.base.ArmTransport` with HMAC frame auth.

    Outgoing lines are signed via :meth:`HmacFramer.sign`.  Incoming lines
    are verified via :meth:`HmacFramer.verify`; frames that fail verification
    are logged and replaced with ``b""`` so the driver's reader loop skips
    them gracefully.

    Args:
        inner: The underlying transport.
        framer: Pre-constructed :class:`HmacFramer`.
    """

    def __init__(self, inner: ArmTransport, framer: HmacFramer) -> None:
        self._inner = inner
        self._framer = framer

    # ------------------------------------------------------------------ #
    # ArmTransport protocol delegation
    # ------------------------------------------------------------------ #

    @property
    def is_connected(self) -> bool:
        """Delegate to the inner transport."""
        return self._inner.is_connected

    async def connect(self) -> None:
        """Delegate to the inner transport."""
        await self._inner.connect()

    async def disconnect(self) -> None:
        """Delegate to the inner transport."""
        await self._inner.disconnect()

    async def write_line(self, data: bytes) -> None:
        """Sign *data* then delegate write to the inner transport.

        Raises:
            ArmDriverError: If *data* is not valid JSON (cannot be signed)
                or if the inner transport raises.
        """
        try:
            signed = self._framer.sign(data)
        except ValueError as exc:
            msg = f"Cannot sign frame: {exc}"
            raise ArmDriverError(msg) from exc
        await self._inner.write_line(signed)

    async def readline(self) -> bytes:
        """Read a line from the inner transport and verify it.

        Returns ``b""`` for empty lines or frames that fail verification,
        letting the driver's reader loop skip them transparently.
        """
        raw = await self._inner.readline()
        if not raw.strip():
            return raw  # empty / keepalive — pass through
        try:
            return self._framer.verify(raw)
        except ValueError as exc:
            _log.warning("transport_auth_failed", error=str(exc))
            return b""


__all__ = ["AuthTransport", "HmacFramer"]
