"""Unit tests for the extended ``ArmConfig`` sub-models.

Covers ``JointLimits``, ``ArmServoConfig``, ``ArmTransportConfig``,
``ArmFirmwareConfig``, and the autosize validator on ``ArmConfig``. The
existing ``test_arm_config.py`` continues to cover the legacy field
surface; this module adds tests for the new structured fields.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from armdroid.config.schema import (
    ArmConfig,
    ArmFirmwareConfig,
    ArmServoConfig,
    ArmSettings,
    ArmTransportConfig,
    JointLimits,
)


class TestJointLimits:
    """Per-joint limits — range validity, gripper-style normalised values."""

    def test_valid_radian_limits(self) -> None:
        lim = JointLimits(min_rad=-1.5708, max_rad=1.5708, max_velocity_rad_s=2.0)
        assert lim.max_rad > lim.min_rad

    def test_valid_normalised_gripper_limits(self) -> None:
        lim = JointLimits(min_rad=0.0, max_rad=1.0, max_velocity_rad_s=5.0)
        assert lim.min_rad == 0.0
        assert lim.max_rad == 1.0

    def test_inverted_range_rejected(self) -> None:
        with pytest.raises(ValidationError, match="max_rad"):
            JointLimits(min_rad=1.0, max_rad=-1.0, max_velocity_rad_s=2.0)

    def test_zero_velocity_rejected(self) -> None:
        with pytest.raises(ValidationError):
            JointLimits(min_rad=-1.0, max_rad=1.0, max_velocity_rad_s=0.0)

    def test_negative_velocity_rejected(self) -> None:
        with pytest.raises(ValidationError):
            JointLimits(min_rad=-1.0, max_rad=1.0, max_velocity_rad_s=-1.0)


class TestArmServoConfig:
    """Servo PWM calibration — pin/pulse range validity."""

    def test_default_pulses(self) -> None:
        servo = ArmServoConfig(pwm_pin=13)
        assert servo.pulse_min_us == 500
        assert servo.pulse_max_us == 2500

    def test_inverted_pulses_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pulse_max_us"):
            ArmServoConfig(pwm_pin=13, pulse_min_us=2500, pulse_max_us=500)

    def test_pin_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ArmServoConfig(pwm_pin=99)


class TestArmTransportConfig:
    """Transport block — defaults, port autodetect, exclusion lists."""

    def test_defaults_use_auto_port(self) -> None:
        cfg = ArmTransportConfig()
        assert cfg.serial_port == "auto"
        assert cfg.serial_baud == 115200
        assert cfg.protocol == "serial"

    def test_max_line_bytes_clamped(self) -> None:
        with pytest.raises(ValidationError):
            ArmTransportConfig(max_line_bytes=32)
        with pytest.raises(ValidationError):
            ArmTransportConfig(max_line_bytes=10_000)

    def test_drain_pings_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            ArmTransportConfig(drain_pings_on_connect=-1)

    def test_exclude_ports_default_empty(self) -> None:
        cfg = ArmTransportConfig()
        assert cfg.exclude_ports == []

    def test_exclude_ports_accept_list(self) -> None:
        cfg = ArmTransportConfig(exclude_ports=["/dev/ttyUSB0", "COM7"])
        assert "COM7" in cfg.exclude_ports


class TestArmFirmwareConfig:
    """Firmware codegen block — interpolator/watchdog/version."""

    def test_defaults(self) -> None:
        cfg = ArmFirmwareConfig()
        assert cfg.interpolator_hz == 50.0
        assert cfg.watchdog_timeout_s == 2.0
        assert cfg.firmware_version.startswith("arm-esp32-")

    def test_interpolator_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            ArmFirmwareConfig(interpolator_hz=500.0)


class TestArmConfigAutosize:
    """Autosize validator pads per-joint defaults when ``dof`` is overridden."""

    def test_default_dof_six_yields_six_entries(self) -> None:
        cfg = ArmConfig()
        assert cfg.dof == 6
        assert len(cfg.joint_limits) == 6
        assert len(cfg.servos) == 6
        assert len(cfg.home_position) == 6

    def test_dof_seven_autosizes_lists(self) -> None:
        cfg = ArmConfig(dof=7)
        assert len(cfg.joint_limits) == 7
        assert len(cfg.servos) == 7
        assert len(cfg.home_position) == 7
        # 7th joint defaults to gripper-style limits
        gripper = cfg.joint_limits[6]
        assert gripper.min_rad == 0.0
        assert gripper.max_rad == 1.0

    def test_explicit_joint_limits_must_match_dof(self) -> None:
        with pytest.raises(ValidationError, match="joint_limits"):
            ArmConfig(
                dof=6,
                joint_limits=[JointLimits(min_rad=-1.0, max_rad=1.0, max_velocity_rad_s=2.0)]
                * 4,  # too few
            )

    def test_home_outside_limits_rejected(self) -> None:
        with pytest.raises(ValidationError, match="home_position"):
            ArmConfig(
                dof=1,
                home_position=[5.0],
                joint_limits=[JointLimits(min_rad=-1.0, max_rad=1.0, max_velocity_rad_s=2.0)],
                servos=[ArmServoConfig(pwm_pin=13)],
            )


class TestArmConfigWatchdogConsistency:
    """firmware.watchdog must exceed transport.keepalive."""

    def test_keepalive_below_watchdog_ok(self) -> None:
        cfg = ArmConfig()  # defaults: watchdog=2.0, keepalive=0.5
        assert cfg.firmware.watchdog_timeout_s > cfg.transport.keepalive_interval_s

    def test_keepalive_above_watchdog_rejected(self) -> None:
        with pytest.raises(ValidationError, match="watchdog_timeout_s"):
            ArmConfig(
                transport=ArmTransportConfig(keepalive_interval_s=5.0),
                firmware=ArmFirmwareConfig(watchdog_timeout_s=1.0),
            )


class TestArmSettingsBackwardsCompat:
    """Top-level ArmSettings still constructs cleanly."""

    def test_root_constructs_with_no_overlay(self) -> None:
        cfg = ArmSettings()
        assert cfg.arm.transport.serial_port == "auto"
        assert cfg.arm.firmware.firmware_version.startswith("arm-esp32-")
        # Legacy top-level transport fields still present:
        assert cfg.arm.serial_port == "COM3"
        assert cfg.arm.serial_baud == 115200
