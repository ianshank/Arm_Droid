"""Wire-frame decoding and pending-reply bookkeeping for the ESP32 JSON driver."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class _PendingReply:
    """A command awaiting its ack/nak from the firmware."""

    future: asyncio.Future[None]
    cmd_name: str


def decode_frame(raw: bytes) -> dict[str, Any]:
    """Decode a single raw UART frame to a message dict.

    Args:
        raw: Raw bytes as returned by ``serial.Serial.readline()``.

    Returns:
        Parsed message dict.

    Raises:
        json.JSONDecodeError: If the payload is not valid JSON.
        UnicodeDecodeError: Propagated from ASCII decoding (rare with
            ``errors='replace'`` but retained for the signature contract).
    """
    text = raw.decode("ascii", errors="replace").strip()
    result: dict[str, Any] = json.loads(text)
    return result


__all__ = ["_PendingReply", "decode_frame"]
