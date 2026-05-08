"""Backwards-compatibility adapters for the v0.1.x driver interface.

These adapters let code written against the **old six-DoF, implicit-timing**
surface continue to work with the new ``ArmDriverProtocol`` without
modification. They emit :class:`DeprecationWarning` on construction so
call-sites can locate and migrate at their own pace.

Scheduled for removal alongside the rest of the compatibility layer in
**v0.4.0**.
"""

from __future__ import annotations

from armdroid.compat.legacy_driver_adapter import LegacyArmDriverAdapter

__all__ = ["LegacyArmDriverAdapter"]
