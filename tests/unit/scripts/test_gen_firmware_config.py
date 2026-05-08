"""Tests for ``scripts/gen_firmware_config.py``.

Cover:

* Header rendering reflects all relevant ArmSettings fields.
* Idempotence — re-running produces the same output.
* ``--check`` mode behaviour: matches => 0, drift => 1, missing => 1.
* CLI smoke: ``main([])`` writes default header.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from armdroid.config.schema import (
    ArmConfig,
    ArmFirmwareConfig,
    ArmServoConfig,
    ArmSettings,
    ArmTransportConfig,
    JointLimits,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "gen_firmware_config.py"


def _load_script_module() -> object:
    """Load scripts/gen_firmware_config.py as a module despite no package."""
    spec = importlib.util.spec_from_file_location("gen_firmware_config", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["gen_firmware_config"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen_module() -> object:
    return _load_script_module()


def _custom_settings() -> ArmSettings:
    """ArmSettings tweaked away from defaults so render values stand out."""
    return ArmSettings(
        arm=ArmConfig(
            dof=6,
            home_position=[0.0] * 6,
            joint_limits=[JointLimits(min_rad=-1.0, max_rad=1.0, max_velocity_rad_s=2.0)] * 6,
            servos=[
                ArmServoConfig(pwm_pin=p, pulse_min_us=600, pulse_max_us=2400)
                for p in [13, 14, 27, 26, 25, 33]
            ],
            transport=ArmTransportConfig(
                serial_baud=230400,
                max_line_bytes=256,
                heartbeat_hz=20.0,
            ),
            firmware=ArmFirmwareConfig(
                interpolator_hz=100.0,
                watchdog_timeout_s=3.0,
                firmware_version="arm-esp32-test-1.2.3",
            ),
        ),
    )


class TestRenderHeader:
    """The rendered header reflects ArmSettings values."""

    def test_namespace_and_pragma(self, gen_module: object) -> None:
        cfg = ArmSettings()
        out = gen_module.render_header(cfg)  # type: ignore[attr-defined]
        assert "#pragma once" in out
        assert "namespace armdroid::firmware::config" in out
        assert out.endswith("}  // namespace armdroid::firmware::config\n")
        # Arduino.h must be guarded so `pio test -e native` works without
        # the embedded toolchain.
        assert "#ifndef UNIT_TEST" in out
        assert "#include <Arduino.h>" in out
        assert "#endif  // UNIT_TEST" in out

    def test_joint_count_matches_dof(self, gen_module: object) -> None:
        cfg = ArmSettings()
        out = gen_module.render_header(cfg)  # type: ignore[attr-defined]
        assert f"constexpr size_t kNumJoints = {cfg.arm.dof};" in out

    def test_pin_array_renders(self, gen_module: object) -> None:
        cfg = _custom_settings()
        out = gen_module.render_header(cfg)  # type: ignore[attr-defined]
        # All six pins should appear as ints in the array initializer
        assert "13, 14, 27, 26, 25, 33" in out

    def test_pulse_calibration_renders(self, gen_module: object) -> None:
        cfg = _custom_settings()
        out = gen_module.render_header(cfg)  # type: ignore[attr-defined]
        assert "kPulseMinUs[kNumJoints] = { 600, 600, 600, 600, 600, 600 }" in out
        assert "kPulseMaxUs[kNumJoints] = { 2400, 2400, 2400, 2400, 2400, 2400 }" in out

    def test_period_conversion_from_hz(self, gen_module: object) -> None:
        # interpolator_hz=100.0 => 10ms period; heartbeat_hz=20.0 => 50ms
        cfg = _custom_settings()
        out = gen_module.render_header(cfg)  # type: ignore[attr-defined]
        assert "kInterpolatorPeriodMs = 10;" in out
        assert "kHeartbeatPeriodMs = 50;" in out
        assert "kWatchdogTimeoutMs = 3000;" in out

    def test_wire_framing_renders(self, gen_module: object) -> None:
        cfg = _custom_settings()
        out = gen_module.render_header(cfg)  # type: ignore[attr-defined]
        assert "kMaxLineBytes = 256;" in out
        assert "kSerialBaud = 230400;" in out

    def test_firmware_version_renders(self, gen_module: object) -> None:
        cfg = _custom_settings()
        out = gen_module.render_header(cfg)  # type: ignore[attr-defined]
        assert 'kFirmwareVersion = "arm-esp32-test-1.2.3";' in out


class TestIdempotence:
    """Re-rendering the same settings yields identical output."""

    def test_idempotent(self, gen_module: object) -> None:
        cfg = _custom_settings()
        first = gen_module.render_header(cfg)  # type: ignore[attr-defined]
        second = gen_module.render_header(cfg)  # type: ignore[attr-defined]
        assert first == second


class TestMainCli:
    """CLI dispatch and ``--check`` mode."""

    def test_main_writes_header(
        self,
        gen_module: object,
        tmp_path: Path,
    ) -> None:
        out_path = tmp_path / "config_generated.h"
        rc = gen_module.main(["--output", str(out_path)])  # type: ignore[attr-defined]
        assert rc == 0
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert "namespace armdroid::firmware::config" in content

    def test_main_idempotent_writes(
        self,
        gen_module: object,
        tmp_path: Path,
    ) -> None:
        out_path = tmp_path / "config_generated.h"
        rc1 = gen_module.main(["--output", str(out_path)])  # type: ignore[attr-defined]
        mtime1 = out_path.stat().st_mtime_ns
        rc2 = gen_module.main(["--output", str(out_path)])  # type: ignore[attr-defined]
        mtime2 = out_path.stat().st_mtime_ns
        assert rc1 == 0
        assert rc2 == 0
        # Second run reports unchanged and does not rewrite the file
        assert mtime1 == mtime2

    def test_check_passes_when_in_sync(
        self,
        gen_module: object,
        tmp_path: Path,
    ) -> None:
        out_path = tmp_path / "config_generated.h"
        gen_module.main(["--output", str(out_path)])  # type: ignore[attr-defined]
        rc = gen_module.main(  # type: ignore[attr-defined]
            ["--output", str(out_path), "--check"]
        )
        assert rc == 0

    def test_check_fails_when_drift(
        self,
        gen_module: object,
        tmp_path: Path,
    ) -> None:
        out_path = tmp_path / "config_generated.h"
        gen_module.main(["--output", str(out_path)])  # type: ignore[attr-defined]
        # Mutate the on-disk header so it no longer matches
        out_path.write_text("// drifted\n", encoding="utf-8")
        rc = gen_module.main(  # type: ignore[attr-defined]
            ["--output", str(out_path), "--check"]
        )
        assert rc == 1

    def test_check_fails_when_missing(
        self,
        gen_module: object,
        tmp_path: Path,
    ) -> None:
        out_path = tmp_path / "config_generated.h"
        rc = gen_module.main(  # type: ignore[attr-defined]
            ["--output", str(out_path), "--check"]
        )
        assert rc == 1
