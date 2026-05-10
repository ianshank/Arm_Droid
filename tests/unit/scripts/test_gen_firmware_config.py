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


class TestSecretsFile:
    """Tests for ``_load_secrets`` and the ``--secrets-file`` CLI flag."""

    def _write_env(self, path: Path, content: str) -> Path:
        secrets = path / "wifi.env"
        secrets.write_text(content, encoding="utf-8")
        return secrets

    def test_load_secrets_with_all_fields(
        self,
        gen_module: object,
        tmp_path: Path,
    ) -> None:
        env_file = self._write_env(
            tmp_path,
            (
                "ARMDROID_WIFI_SSID=MyNetwork\n"
                "ARMDROID_WIFI_PASS=hunter2\n"
                f"ARMDROID_HMAC_KEY={'ab' * 16}\n"
                "ARMDROID_TCP_PORT=4001\n"
            ),
        )
        secrets = gen_module._load_secrets(env_file)  # type: ignore[attr-defined]
        assert secrets.wifi_ssid == "MyNetwork"
        assert secrets.wifi_password == "hunter2"  # noqa: S105 — test fixture, not a real secret
        assert secrets.hmac_key_hex == "ab" * 16
        assert secrets.tcp_port == 4001
        assert secrets.wifi_enabled is True
        assert secrets.hmac_enabled is True

    def test_load_secrets_strips_quotes(
        self,
        gen_module: object,
        tmp_path: Path,
    ) -> None:
        env_file = self._write_env(
            tmp_path,
            "ARMDROID_WIFI_SSID=\"Quoted Network\"\nARMDROID_WIFI_PASS='s3cr3t'\n",
        )
        secrets = gen_module._load_secrets(env_file)  # type: ignore[attr-defined]
        assert secrets.wifi_ssid == "Quoted Network"
        assert secrets.wifi_password == "s3cr3t"  # noqa: S105 — test fixture, not a real secret

    def test_load_secrets_ignores_comments_and_blank_lines(
        self,
        gen_module: object,
        tmp_path: Path,
    ) -> None:
        env_file = self._write_env(
            tmp_path,
            "# This is a comment\n\nARMDROID_WIFI_SSID=Net1\n",
        )
        secrets = gen_module._load_secrets(env_file)  # type: ignore[attr-defined]
        assert secrets.wifi_ssid == "Net1"

    def test_load_secrets_empty_file_returns_disabled(
        self,
        gen_module: object,
        tmp_path: Path,
    ) -> None:
        env_file = self._write_env(tmp_path, "")
        secrets = gen_module._load_secrets(env_file)  # type: ignore[attr-defined]
        assert secrets.wifi_enabled is False
        assert secrets.hmac_enabled is False

    def test_load_secrets_invalid_hmac_hex_exits(
        self,
        gen_module: object,
        tmp_path: Path,
    ) -> None:
        env_file = self._write_env(
            tmp_path,
            "ARMDROID_HMAC_KEY=not_hex_at_all\n",
        )
        with pytest.raises(SystemExit) as exc_info:
            gen_module._load_secrets(env_file)  # type: ignore[attr-defined]
        assert exc_info.value.code == 1

    def test_load_secrets_hmac_key_too_short_exits(
        self,
        gen_module: object,
        tmp_path: Path,
    ) -> None:
        env_file = self._write_env(
            tmp_path,
            "ARMDROID_HMAC_KEY=deadbeef\n",  # 8 hex chars = 4 bytes < 16 required
        )
        with pytest.raises(SystemExit) as exc_info:
            gen_module._load_secrets(env_file)  # type: ignore[attr-defined]
        assert exc_info.value.code == 1

    def test_load_secrets_hmac_key_odd_length_exits(
        self,
        gen_module: object,
        tmp_path: Path,
    ) -> None:
        """Odd-length hex strings cannot be decoded by bytes.fromhex()."""
        env_file = self._write_env(
            tmp_path,
            "ARMDROID_HMAC_KEY=" + "ab" * 16 + "c\n",  # 33 hex chars (odd)
        )
        with pytest.raises(SystemExit) as exc_info:
            gen_module._load_secrets(env_file)  # type: ignore[attr-defined]
        assert exc_info.value.code == 1

    def test_load_secrets_bad_tcp_port_exits(
        self,
        gen_module: object,
        tmp_path: Path,
    ) -> None:
        env_file = self._write_env(
            tmp_path,
            "ARMDROID_TCP_PORT=notanumber\n",
        )
        with pytest.raises(SystemExit) as exc_info:
            gen_module._load_secrets(env_file)  # type: ignore[attr-defined]
        assert exc_info.value.code == 1

    def test_render_header_with_secrets_enables_wifi(
        self,
        gen_module: object,
        tmp_path: Path,
    ) -> None:
        env_file = self._write_env(
            tmp_path,
            (
                "ARMDROID_WIFI_SSID=AcmeNet\n"
                "ARMDROID_WIFI_PASS=p@ssw0rd\n"
                f"ARMDROID_HMAC_KEY={'cd' * 16}\n"
                "ARMDROID_TCP_PORT=3001\n"
            ),
        )
        secrets = gen_module._load_secrets(env_file)  # type: ignore[attr-defined]
        cfg = ArmSettings()
        out = gen_module.render_header(cfg, secrets=secrets)  # type: ignore[attr-defined]
        assert "kWiFiEnabled = true;" in out
        assert 'kWiFiSSID = "AcmeNet";' in out
        assert 'kWiFiPassword = "p@ssw0rd";' in out
        assert "kHmacEnabled = true;" in out
        assert f'kHmacKeyHex = "{"cd" * 16}";' in out

    def test_render_header_without_secrets_disables_wifi(
        self,
        gen_module: object,
    ) -> None:
        cfg = ArmSettings()
        out = gen_module.render_header(cfg)  # type: ignore[attr-defined]
        assert "kWiFiEnabled = false;" in out
        assert "kHmacEnabled = false;" in out

    def test_render_header_c_string_escaping(
        self,
        gen_module: object,
        tmp_path: Path,
    ) -> None:
        """SSID with backslash and double-quote must be properly escaped."""
        env_file = self._write_env(
            tmp_path,
            'ARMDROID_WIFI_SSID=Net\\"Work\nARMDROID_WIFI_PASS=x\n',
        )
        secrets = gen_module._load_secrets(env_file)  # type: ignore[attr-defined]
        cfg = ArmSettings()
        out = gen_module.render_header(cfg, secrets=secrets)  # type: ignore[attr-defined]
        # The backslash and double-quote in the SSID must be escaped in the C string.
        # Net\"Work in C (3 chars: \, \, \", ") represents Net\"Work literally.
        assert r'kWiFiSSID = "Net\\\"Work";' in out

    def test_main_with_secrets_file_writes_wifi_constants(
        self,
        gen_module: object,
        tmp_path: Path,
    ) -> None:
        env_file = self._write_env(
            tmp_path,
            (
                "ARMDROID_WIFI_SSID=TestNet\n"
                "ARMDROID_WIFI_PASS=pass1\n"
                f"ARMDROID_HMAC_KEY={'ef' * 16}\n"
            ),
        )
        out_path = tmp_path / "config_generated.h"
        rc = gen_module.main(  # type: ignore[attr-defined]
            ["--output", str(out_path), "--secrets-file", str(env_file)]
        )
        assert rc == 0
        content = out_path.read_text(encoding="utf-8")
        assert "kWiFiEnabled = true;" in content
        assert 'kWiFiSSID = "TestNet";' in content

    def test_main_secrets_file_missing_exits_nonzero(
        self,
        gen_module: object,
        tmp_path: Path,
    ) -> None:
        out_path = tmp_path / "config_generated.h"
        missing = tmp_path / "no_such.env"
        rc = gen_module.main(  # type: ignore[attr-defined]
            ["--output", str(out_path), "--secrets-file", str(missing)]
        )
        assert rc != 0
