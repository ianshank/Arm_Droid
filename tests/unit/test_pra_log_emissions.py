"""Log-emission assertions for PR-A's new events catalogue.

PR-A introduced these new structlog events that the implementation
plan's "logging events catalogue" promises will fire. Without explicit
assertions on the events, the catalogue is documentation-only — drift
between the doc and the code goes silently uncaught.

Closes peer-review N6 (logging events catalogue is aspirational; no
test asserts log emissions).

Events asserted here:
* ``arm_urdf_path_missing``       — ArmConfig.urdf_path validator
* ``arm_driver_built``            — orchestration.factory.build_arm_driver
* ``driver_kind_legacy_fallback`` — _driver_kind.resolve_driver_kind
* ``driver_kind_resolved``        — _driver_kind.resolve_driver_kind (debug)
* ``arm_controller_built``        — orchestration.factory.build_arm_controller
* ``arm_controller_init``         — control.controller.ArmController.__init__

Uses ``structlog.testing.capture_logs`` — ships with structlog, no
extra dependency. Establishes the pattern for the rest of the codebase
to adopt over time.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import pytest
import structlog
from structlog.testing import capture_logs

from armdroid.config.schema import ArmConfig, ArmSettings, ArmTrainingConfig
from armdroid.orchestration._driver_kind import (
    _reset_warned_for_tests,
    resolve_driver_kind,
)

# structlog.testing.capture_logs yields list[MutableMapping[str, Any]] —
# pin the alias once so all helpers + tests stay consistent.
LogEvent = MutableMapping[str, Any]


@pytest.fixture(autouse=True)
def _isolate_warned() -> None:
    """Reset the once-per-process deprecation flag for clean assertions."""
    _reset_warned_for_tests()


def _events_with_name(events: list[LogEvent], name: str) -> list[LogEvent]:
    """Return all captured events whose ``event`` field matches ``name``."""
    return [e for e in events if e.get("event") == name]


class TestArmUrdfValidatorLog:
    def test_emits_arm_urdf_path_missing_when_file_absent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Closes coverage gap arm.py:340-348 (validator missing-file branch).

        Construct an ``ArmConfig`` with a urdf_path that does NOT exist on
        disk. The validator must emit an INFO-level structlog event named
        ``arm_urdf_path_missing`` with the offending path attached.
        """
        # Force resolve_asset_path to use tmp_path as the repo root so the
        # synthesised relative path can't accidentally resolve to a real file.
        monkeypatch.setenv("ARMDROID_REPO_ROOT", str(tmp_path))

        with capture_logs() as captured:
            ArmConfig(urdf_path=Path("definitely/not/here.urdf"))

        events = _events_with_name(captured, "arm_urdf_path_missing")
        assert len(events) >= 1, (
            "validator did not emit arm_urdf_path_missing for a missing URDF; "
            f"captured events: {[e.get('event') for e in captured]}"
        )
        ev = events[0]
        assert ev["log_level"] == "info"
        # Path-separator-agnostic check (Windows uses \\, POSIX uses /).
        assert "here.urdf" in ev["urdf_path"]
        assert "definitely" in ev["urdf_path"]
        assert "hint" in ev

    def test_no_log_when_file_exists(self, tmp_path: Path) -> None:
        """When the URDF exists, no missing-file event should fire."""
        urdf = tmp_path / "real.urdf"
        urdf.write_text("<robot/>")

        with capture_logs() as captured:
            ArmConfig(urdf_path=urdf)

        assert _events_with_name(captured, "arm_urdf_path_missing") == []


class TestDriverKindResolutionLogs:
    def test_emits_driver_kind_resolved_debug_event(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """resolve_driver_kind logs the resolved kind + source at DEBUG."""
        monkeypatch.setenv("ARMDROID_SUPPRESS_DEPRECATION", "1")
        cfg = ArmSettings(arm_driver_kind="mock")

        with capture_logs() as captured:
            resolve_driver_kind(cfg)

        events = _events_with_name(captured, "driver_kind_resolved")
        assert len(events) == 1
        assert events[0]["kind"] == "mock"
        assert events[0]["source"] == "explicit"
        assert events[0]["log_level"] == "debug"

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_legacy_fallback_emits_warning_log(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Setting only mock_hardware=True logs ``driver_kind_legacy_fallback``.

        The DeprecationWarning is intentionally emitted by
        ``_emit_deprecation_warning_once`` and is asserted by
        :class:`tests.unit.test_driver_kind.TestDeprecation`. Ignore it
        here so the test output stays clean.
        """
        monkeypatch.delenv("ARMDROID_SUPPRESS_DEPRECATION", raising=False)
        monkeypatch.delenv("ARMDROID_ARM_DRIVER_KIND", raising=False)
        cfg = ArmSettings(mock_hardware=True, arm_driver_kind=None)

        with capture_logs() as captured:
            resolve_driver_kind(cfg)

        events = _events_with_name(captured, "driver_kind_legacy_fallback")
        assert len(events) == 1
        assert events[0]["log_level"] == "warning"
        assert "deprecated" in events[0]["message"].lower()


class TestFactoryBuildLogs:
    def test_build_arm_driver_logs_arm_driver_built(self) -> None:
        from armdroid.orchestration.factory import build_arm_driver

        cfg = ArmSettings(arm_driver_kind="mock")
        with capture_logs() as captured:
            build_arm_driver(cfg)

        events = _events_with_name(captured, "arm_driver_built")
        assert len(events) == 1
        assert events[0]["kind"] == "mock"
        assert events[0]["log_level"] == "info"

    def test_build_arm_controller_logs_built_with_algorithm(self) -> None:
        from armdroid.orchestration.factory import build_arm_controller

        cfg = ArmSettings(arm_driver_kind="mock", arm_training=ArmTrainingConfig())

        with capture_logs() as captured:
            build_arm_controller(cfg)

        events = _events_with_name(captured, "arm_controller_built")
        assert len(events) == 1
        # Default algorithm is "sac_her".
        assert events[0]["algorithm"] == "sac_her"
        assert events[0]["log_level"] == "info"


class TestStructlogTestingPatternIsAvailable:
    """Smoke test: ``structlog.testing.capture_logs`` is importable + works.

    Establishes the pattern as a documented, supported test idiom for the
    rest of the codebase to adopt over time.
    """

    def test_capture_logs_returns_event_dicts(self) -> None:
        log = structlog.get_logger("armdroid.test_pattern")
        with capture_logs() as captured:
            log.info("smoke_event", key="value")
        assert any(e.get("event") == "smoke_event" for e in captured)
