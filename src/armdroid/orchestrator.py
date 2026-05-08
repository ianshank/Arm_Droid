"""Backwards-compatibility re-export for the legacy ``armdroid.orchestrator`` path.

The canonical home is :mod:`armdroid.orchestration.orchestrator`. New code
should import from :mod:`armdroid` (public façade) or
:mod:`armdroid.orchestration` directly. This shim is preserved for the
v0.2.x line and scheduled for removal in v0.4.0.

The private helper :func:`_step_args_to_target` is re-exported for tests
that exercise it directly; it remains internal and may be relocated in a
later phase without notice.
"""

from __future__ import annotations

from armdroid.orchestration.orchestrator import ArmOrchestrator, _step_args_to_target

__all__ = ["ArmOrchestrator", "_step_args_to_target"]
