"""Typed unit wrappers — Phase 4 expansion seam.

This module reserves the public namespace ``armdroid.domain.units`` for
typed scalar wrappers (``Radians``, ``Seconds``, ``Hz``, ``Newtons``) that
will replace bare ``float`` at API boundaries in the v0.2.x line. The
wrappers are deliberately empty in v0.2.0 to preserve the import path
without yet forcing a typing migration through the codebase.

See ``docs/architecture/PHASES.md`` for the timeline and
``ADR-0001-enterprise-layering`` for the rationale.
"""

from __future__ import annotations

from typing import NewType

# ``NewType`` aliases provide nominal typing without runtime overhead. They
# are recognised by mypy/pyright as distinct from ``float`` while remaining
# transparent to numerics. Phase 4 will introduce conversion helpers and
# adopt them at controller / driver boundaries.
Radians = NewType("Radians", float)
Seconds = NewType("Seconds", float)
Hz = NewType("Hz", float)
Newtons = NewType("Newtons", float)
Meters = NewType("Meters", float)

__all__ = ["Hz", "Meters", "Newtons", "Radians", "Seconds"]
