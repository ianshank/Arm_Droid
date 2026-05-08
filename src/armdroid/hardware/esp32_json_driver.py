"""Backwards-compat shim for armdroid.hardware.esp32_json_driver.

Re-exports from :mod:`armdroid.hardware.esp32`. This module is preserved as a
permanent re-export shim so that every existing import path keeps working
without modification through to v0.4.0, per ADR-0001.

The canonical implementation lives in :mod:`armdroid.hardware.esp32.driver`.
"""

from __future__ import annotations

from armdroid.hardware.esp32.driver import (
    _FIRST_STATE_POLL_INTERVAL_S as _FIRST_STATE_POLL_INTERVAL_S,
)
from armdroid.hardware.esp32.driver import (
    _KEEPALIVE_POLL_FLOOR_S as _KEEPALIVE_POLL_FLOOR_S,
)

# Re-export the public class and module-level constants so that code
# probing ``armdroid.hardware.esp32_json_driver._KEEPALIVE_POLL_FLOOR_S``
# (e.g. regression tests) continues to work without modification.
# Using ``X as X`` syntax marks each name as an intentional re-export.
from armdroid.hardware.esp32.driver import (
    Esp32JsonDriver as Esp32JsonDriver,
)

__all__ = ["Esp32JsonDriver"]
