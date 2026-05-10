"""Process-wide AppLauncher state for Isaac Sim.

Isaac Sim's Kit cannot be launched twice in the same Python process —
the second call produces an inscrutable Kit double-init traceback.
This module hosts a singleton flag so every Isaac-runtime entry point
(``IsaacSimDriver``, ``SoArmReachIsaacEnv``, future agents) shares the
same view of "is Kit running?".

Why module-level state and not a class attribute? Python imports modules
exactly once per process; a module-level flag is the simplest way to
share state across packages without coupling the consumers.

The :func:`is_app_launched` probe also tries to detect a live
``omni.kit.app`` instance — this catches the case where Kit was launched
out-of-band (e.g. user code constructed ``AppLauncher`` directly before
calling our entry points).
"""

from __future__ import annotations

_app_launched: bool = False


def is_app_launched() -> bool:
    """Return ``True`` if Kit has been booted in this process.

    Probes the live ``omni.kit.app`` where possible so out-of-band
    launches (someone constructed ``AppLauncher`` directly before us)
    are detected. Falls back to the local flag when the probe is
    unavailable (default install without ``[isaac]``).
    """
    global _app_launched
    if _app_launched:
        return True
    try:  # pragma: no cover - probe only available with [isaac] extra
        import omni.kit.app

        if omni.kit.app.get_app() is not None:
            _app_launched = True
            return True
    except Exception as exc:
        # Probe failure is expected on default installs (no omni). Log
        # at DEBUG so live failures (Kit broken mid-run) stay visible.
        from armdroid.logging.setup import get_logger

        get_logger(__name__).debug("isaac_app_probe_failed", error=str(exc))
    return False


def mark_launched() -> None:
    """Mark Kit as launched. Called by the first successful boot."""
    global _app_launched
    _app_launched = True


def reset_for_tests() -> None:
    """Reset the launch flag. Tests only — never call from production code."""
    global _app_launched
    _app_launched = False


__all__ = ["is_app_launched", "mark_launched", "reset_for_tests"]
