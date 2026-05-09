"""Backwards-compatible driver-kind resolution.

The legacy ``ArmSettings.mock_hardware: bool`` field is preserved so
that every existing YAML, env var, and test continues to work unchanged.
The new ``ArmSettings.arm_driver_kind: Literal[...] | None`` field is
the source of truth for new code.

Precedence:
    1. ``cfg.arm_driver_kind`` when not None — explicit wins, no warning.
    2. ``cfg.mock_hardware == True`` — emits once-per-process deprecation
       warning (unless ``ARMDROID_SUPPRESS_DEPRECATION=1``) and returns
       ``"mock"``.
    3. Otherwise — returns the legacy default ``"esp32"`` without warning.

PR-A keeps the literal narrow at ``["mock", "esp32"]``. PR-B widens it
to include ``"isaac_sim"`` alongside the registry registration so an
explicit ``arm_driver_kind="isaac_sim"`` cannot KeyError before the
driver lands.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Final, Literal

from armdroid.logging.setup import get_logger

if TYPE_CHECKING:
    from armdroid.config.schema import ArmSettings

DriverKind = Literal["mock", "esp32"]
# v2: PR-A keeps the literal narrow. PR-B (B.2) widens this to include
# "isaac_sim" alongside the registry registration so an explicit set of
# arm_driver_kind="isaac_sim" cannot crash with KeyError before B merges.

_DEPRECATION_MSG: Final[str] = (
    "ArmSettings.mock_hardware is deprecated; set arm_driver_kind="
    '"mock" or "esp32" instead. mock_hardware will be removed in '
    "armdroid v0.4.0."
)
_SUPPRESS_ENV: Final[str] = "ARMDROID_SUPPRESS_DEPRECATION"
_log = get_logger(__name__)
_warned: bool = False


def _reset_warned_for_tests() -> None:
    """Reset the once-per-process flag — tests only."""
    global _warned
    _warned = False


def _emit_deprecation_warning_once() -> None:
    """Emit the deprecation log + warnings.warn at most once per process."""
    global _warned
    if _warned:
        return
    _warned = True
    if os.environ.get(_SUPPRESS_ENV) == "1":
        return
    _log.warning("driver_kind_legacy_fallback", message=_DEPRECATION_MSG)
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=3)


def resolve_driver_kind(cfg: ArmSettings) -> DriverKind:
    """Resolve the driver kind respecting the legacy mock_hardware bool.

    Args:
        cfg: Root settings.

    Returns:
        One of ``"mock"`` or ``"esp32"`` (PR-B will extend the union).
    """
    if cfg.arm_driver_kind is not None:
        kind: DriverKind = cfg.arm_driver_kind
        _log.debug("driver_kind_resolved", kind=kind, source="explicit")
        return kind
    if cfg.mock_hardware:
        _emit_deprecation_warning_once()
        _log.debug("driver_kind_resolved", kind="mock", source="legacy_bool")
        return "mock"
    _log.debug("driver_kind_resolved", kind="esp32", source="legacy_default")
    return "esp32"


# Public surface only — underscore-prefixed names are accessed directly
# from the module by tests (see tests/unit/test_driver_kind.py).
__all__ = ["DriverKind", "resolve_driver_kind"]
