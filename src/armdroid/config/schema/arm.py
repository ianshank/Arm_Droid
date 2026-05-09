"""Robot arm hardware configuration.

Contains :class:`JointLimits`, :class:`ArmServoConfig`,
:class:`ArmTransportConfig`, :class:`ArmFirmwareConfig`, and the top-level
:class:`ArmConfig` that aggregates them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from armdroid.config.paths import SO101_URDF_REL, resolve_asset_path
from armdroid.logging.setup import get_logger

_log = get_logger(__name__)


class JointLimits(BaseModel):
    """Per-joint position and velocity limits.

    Used by both the host-side validator (in driver implementations) and
    the firmware codegen script. Values are radians for the rotational
    joints and normalised ``[0, 1]`` for the gripper, where ``0`` is fully
    open and ``1`` is fully closed.
    """

    min_rad: float = Field(
        ...,
        description="Minimum joint position (radians, or [0, 1] for gripper).",
    )
    max_rad: float = Field(
        ...,
        description="Maximum joint position (radians, or [0, 1] for gripper).",
    )
    max_velocity_rad_s: float = Field(
        ...,
        gt=0.0,
        description="Maximum commanded joint velocity (rad/s). Driver rejects "
        "moves whose required speed (|target - start| / duration) exceeds this.",
    )

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.max_rad <= self.min_rad:
            msg = f"max_rad ({self.max_rad}) must exceed min_rad ({self.min_rad})"
            raise ValueError(msg)
        return self


class ArmServoConfig(BaseModel):
    """Per-servo PWM calibration for the firmware codegen.

    These values are written into ``firmware/arm_esp32/src/config_generated.h``
    by ``scripts/gen_firmware_config.py`` so host and firmware share one
    source of truth for hardware tunables.
    """

    pwm_pin: int = Field(
        ...,
        ge=0,
        le=39,
        description="ESP32 GPIO pin number wired to the servo signal line.",
    )
    pulse_min_us: int = Field(
        500,
        ge=200,
        le=3000,
        description="Servo pulse-width at the joint's minimum mechanical position (microseconds).",
    )
    pulse_max_us: int = Field(
        2500,
        ge=200,
        le=3000,
        description="Servo pulse-width at the joint's maximum mechanical position (microseconds).",
    )

    @model_validator(mode="after")
    def _check_pulse_order(self) -> Self:
        if self.pulse_max_us <= self.pulse_min_us:
            msg = (
                f"pulse_max_us ({self.pulse_max_us}) must exceed pulse_min_us ({self.pulse_min_us})"
            )
            raise ValueError(msg)
        return self


class ArmTransportConfig(BaseModel):
    """Transport-layer settings for the real arm driver.

    Mirrors the wire-protocol assumptions in ``firmware/arm_esp32/PROTOCOL.md``.
    All numeric thresholds the driver consults at runtime (line size cap,
    drain-ping count, first-state-wait budget) live here so nothing in the
    driver implementation is hardcoded.
    """

    protocol: Literal["serial"] = Field(
        default="serial",
        description="Wire protocol used to talk to the arm microcontroller. "
        "'serial' = newline-delimited JSON over UART (PROTOCOL.md). "
        "Future variants (e.g. 'websocket') do not require a schema change.",
    )
    serial_port: str = Field(
        default="auto",
        description="Serial device path (e.g. '/dev/ttyUSB1', 'COM3'). "
        "Set to 'auto' to enumerate USB serial ports and probe each one "
        "for the firmware boot signature.",
    )
    serial_baud: int = Field(
        default=115200,
        gt=0,
        description="Serial baud rate. Must match firmware kSerialBaud.",
    )
    connect_timeout_s: float = Field(
        default=5.0,
        gt=0.0,
        description="Maximum time to wait for the firmware to respond on connect.",
    )
    command_timeout_s: float = Field(
        default=0.25,
        gt=0.0,
        description="Per-command ack timeout. Tight bound because the "
        "orchestrator runs at 30 Hz; loose enough to absorb USB jitter.",
    )
    heartbeat_hz: float = Field(
        default=10.0,
        gt=0.0,
        description="Expected firmware state-broadcast rate. Codegen mirrors "
        "this into firmware kHeartbeatPeriodMs.",
    )
    max_line_bytes: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Hard cap on a single JSON wire frame. Mirrors firmware "
        "kMaxLineBytes; longer lines are dropped.",
    )
    drain_pings_on_connect: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of ping/ack round-trips issued during connect to "
        "drain stale boot chatter from the UART before motion commands.",
    )
    first_state_wait_s: float = Field(
        default=0.20,
        gt=0.0,
        description="Time the driver waits for the first state heartbeat after "
        "issuing get_state at startup (before declaring an ArmDriverError).",
    )
    keepalive_interval_s: float = Field(
        default=0.5,
        gt=0.0,
        description="If no motion command has been sent for this long, the "
        "driver issues a ping to keep the firmware watchdog satisfied. "
        "Must be < firmware.watchdog_timeout_s.",
    )
    exclude_ports: list[str] = Field(
        default_factory=list,
        description="Serial ports to skip during 'auto' discovery. Use to "
        "avoid binding to other ESP32 devices (e.g. a rover controller).",
    )
    usb_vid_pid_hints: list[str] = Field(
        default_factory=list,
        description="Optional 'VID:PID' hex strings to prefer during 'auto' "
        "discovery. Empty list means probe all available ports.",
    )
    autodetect_probe_concurrency: int = Field(
        default=4,
        ge=1,
        le=32,
        description="How many candidate ports to probe in parallel during 'auto' discovery.",
    )


class ArmFirmwareConfig(BaseModel):
    """Firmware-side compile-time settings consumed by the codegen.

    Values from this block are written into
    ``firmware/arm_esp32/src/config_generated.h`` by
    ``scripts/gen_firmware_config.py``. None are read by the host at runtime.
    """

    interpolator_hz: float = Field(
        default=50.0,
        gt=0.0,
        le=200.0,
        description="Servo refresh rate. Codegen converts to "
        "kInterpolatorPeriodMs = 1000 / interpolator_hz.",
    )
    watchdog_timeout_s: float = Field(
        default=2.0,
        gt=0.0,
        description="If no host command arrives for this duration, firmware "
        "auto-latches e-stop. Must exceed transport.keepalive_interval_s.",
    )
    firmware_version: str = Field(
        default="arm-esp32-1.0.0",
        description="Version string emitted in the boot evt frame. Bump on wire-protocol changes.",
    )


def _default_joint_limits_6dof() -> list[JointLimits]:
    """Conservative defaults for the legacy 6-DoF arm (no gripper)."""
    return [
        JointLimits(min_rad=-1.5708, max_rad=1.5708, max_velocity_rad_s=2.0),
        JointLimits(min_rad=-0.5, max_rad=1.5708, max_velocity_rad_s=1.5),
        JointLimits(min_rad=-1.5708, max_rad=1.5708, max_velocity_rad_s=2.0),
        JointLimits(min_rad=-1.5708, max_rad=1.5708, max_velocity_rad_s=3.0),
        JointLimits(min_rad=-1.5708, max_rad=1.5708, max_velocity_rad_s=4.0),
        JointLimits(min_rad=-1.5708, max_rad=1.5708, max_velocity_rad_s=4.0),
    ]


def _default_servos_6dof() -> list[ArmServoConfig]:
    """ESP32 DevKitC pin map for joints 0-5 (no gripper at 6 DoF)."""
    pins = [13, 14, 27, 26, 25, 33]
    return [ArmServoConfig(pwm_pin=p) for p in pins]


class ArmConfig(BaseModel):
    """Robot arm hardware configuration.

    Backwards-compatible top-level fields (``dof``, ``home_position``,
    ``serial_port``, ``serial_baud``, ``command_timeout_s``,
    ``gripper_type``, ``max_joint_velocity_rads``, ``max_joint_torque_nm``)
    are preserved for existing tests and YAML files. New code should read
    from the structured sub-models (``transport``, ``joint_limits``,
    ``servos``, ``firmware``) instead.

    The 7th joint (gripper) lands in commit 7 of the ESP32 integration; until
    then ``dof`` defaults to 6 and the gripper-specific fields are not
    populated.
    """

    urdf_path: Path = Field(
        SO101_URDF_REL,
        description=(
            "Path to robot arm URDF file. Repo-relative inputs are resolved "
            "to absolute via armdroid.config.paths.resolve_asset_path() at "
            "construction time, so consumers see a stable absolute path "
            "regardless of CWD. Loaders should branch on ``Path.is_file()`` "
            "so optional / mock paths still construct."
        ),
    )
    dof: int = Field(6, gt=0, le=12, description="Degrees of freedom")
    gripper_type: Literal["parallel", "suction", "soft"] = Field(
        "parallel",
        description="End-effector gripper type",
    )
    max_joint_velocity_rads: float = Field(2.0, gt=0, description="Max joint velocity (rad/s)")
    max_joint_torque_nm: float = Field(5.0, gt=0, description="Max joint torque (Nm)")
    home_position: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        description="Home joint angles (rad) — length must match dof",
    )
    home_duration_s: float = Field(
        default=2.0,
        gt=0.0,
        description="Duration over which the firmware interpolates from any "
        "pose back to home_position. Used by primitives.home().",
    )
    transit_duration_s: float = Field(
        default=2.0,
        gt=0.0,
        description="Duration over which the firmware interpolates a transit "
        "primitive (free-space joint move). Drives "
        "ActionPrimitives.transit() in the modern motion path.",
    )
    grasp_duration_s: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Duration over which the firmware interpolates the approach phase "
            "of the grasp primitive on the modern (gripper-as-joint) path. "
            "Gripper open/close itself is a separate one-frame write dispatched "
            "at the same duration via send_joint_positions."
        ),
    )
    place_duration_s: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Duration over which the firmware interpolates the approach phase "
            "of the place/deposit primitive on the modern (gripper-as-joint) path. "
            "Gripper open itself is a separate one-frame write dispatched "
            "at the same duration via send_joint_positions."
        ),
    )
    # Legacy top-level transport fields. Preferred location is
    # ``transport.*`` below; these stay for backwards compatibility.
    serial_port: str = Field("COM3", description="Serial port for arm controller (legacy)")
    serial_baud: int = Field(115200, gt=0, description="Serial baud rate (legacy)")
    command_timeout_s: float = Field(
        1.0, gt=0, description="Command response timeout, seconds (legacy)"
    )
    # Structured sub-models — single source of truth for new code.
    joint_limits: list[JointLimits] = Field(
        default_factory=_default_joint_limits_6dof,
        description="Per-joint position and velocity limits. Length must match dof.",
    )
    servos: list[ArmServoConfig] = Field(
        default_factory=_default_servos_6dof,
        description="Per-servo PWM calibration (firmware codegen). Length must match dof.",
    )
    transport: ArmTransportConfig = Field(
        default_factory=ArmTransportConfig,
        description="Transport-layer settings for the real driver "
        "(serial port, baud, timeouts, line-size cap, port autodetect).",
    )
    firmware: ArmFirmwareConfig = Field(
        default_factory=ArmFirmwareConfig,
        description="Firmware compile-time settings (consumed by codegen).",
    )
    workspace_envelope_radius_m: float = Field(
        default=0.30,
        gt=0.0,
        description="Maximum end-effector distance from the base origin. "
        "Used by the safety envelope check.",
    )
    self_collision_margin_m: float = Field(
        default=0.02,
        ge=0.0,
        description="Minimum permitted distance between any two non-adjacent "
        "links. Used by the self-collision check.",
    )

    @field_validator("urdf_path", mode="after")
    @classmethod
    def _resolve_and_warn_if_urdf_missing(cls, v: Path) -> Path:
        """Resolve to absolute and emit an INFO log when the URDF is missing.

        Does not raise — mock-hardware test setups and CI runners without
        the vendored asset tree must keep constructing
        :class:`ArmSettings`. Loaders downstream branch on
        :meth:`Path.is_file` and produce a real error only if a real load
        is attempted.

        Returns the resolved absolute path so callers see a CWD-stable
        value regardless of where the YAML / env var was authored.
        """
        resolved = resolve_asset_path(v)
        if not resolved.is_file():
            _log.info(
                "arm_urdf_path_missing",
                urdf_path=str(v),
                resolved=str(resolved),
                hint="vendor SO101 assets or override arm.urdf_path",
            )
        return resolved

    @model_validator(mode="before")
    @classmethod
    def _mirror_legacy_transport(cls, data: object) -> object:
        """Mirror legacy top-level transport fields into ``transport.*``.

        Old configs may set ``arm.serial_port``, ``arm.serial_baud``, or
        ``arm.command_timeout_s`` at the top level.  When no explicit
        ``transport`` sub-config is present those values are forwarded into
        the transport block so the real driver still picks them up.
        """
        if not isinstance(data, dict):
            return data
        if "transport" in data:
            return data  # explicit transport block wins; nothing to mirror
        transport: dict[str, object] = {}
        if "serial_port" in data:
            transport["serial_port"] = data["serial_port"]
        if "serial_baud" in data:
            transport["serial_baud"] = data["serial_baud"]
        if "command_timeout_s" in data:
            transport["command_timeout_s"] = data["command_timeout_s"]
        if transport:
            data["transport"] = transport
        return data

    @model_validator(mode="before")
    @classmethod
    def _autosize_per_joint_lists(cls, data: object) -> object:
        """Auto-size per-joint default lists when ``dof`` is overridden.

        If the user sets ``dof`` to a value other than 6 *and* does not
        supply explicit ``joint_limits`` / ``servos`` / ``home_position``,
        the per-joint defaults below are stretched/truncated so the
        resulting config is still valid. Explicit user-provided lists are
        passed through unchanged so genuine length mismatches still
        surface as validation errors in the after-mode validator.
        """
        if not isinstance(data, dict):
            return data
        dof = data.get("dof", 6)
        try:
            dof_int = int(dof)
        except (TypeError, ValueError):
            return data
        if dof_int <= 0:
            return data
        # joint_limits — only inject default if absent
        if "joint_limits" not in data:
            base = _default_joint_limits_6dof()
            if dof_int <= len(base):
                limits = base[:dof_int]
            else:
                gripper = JointLimits(min_rad=0.0, max_rad=1.0, max_velocity_rad_s=5.0)
                extra = [gripper] * (dof_int - len(base))
                limits = base + extra
            data["joint_limits"] = [
                lim.model_dump() if isinstance(lim, JointLimits) else lim for lim in limits
            ]
        # servos — only inject default if absent
        if "servos" not in data:
            # ESP32 strapping pins (GPIO 0, 2, 5, 12, 15) are excluded — they
            # affect boot mode and flash voltage and must be free at reset.
            # Pin 4 and 16-23 are general-purpose output-safe.
            base_pins = [13, 14, 27, 26, 25, 33, 32, 4, 16, 17, 18, 19]
            chosen = (
                base_pins[:dof_int]
                if dof_int <= len(base_pins)
                else base_pins + [21] * (dof_int - len(base_pins))
            )
            data["servos"] = [{"pwm_pin": p} for p in chosen]
        # home_position — only inject default if absent
        if "home_position" not in data:
            data["home_position"] = [0.0] * dof_int
        return data

    @property
    def gripper_joint_index(self) -> int | None:
        """Index of the gripper joint in the joint vector, or ``None``.

        Returns the index of the gripper joint when this configuration
        models the gripper as an explicit protocol joint (typically
        ``dof - 1`` when ``dof >= 7``). When ``dof <= 6`` the gripper is
        external to the joint protocol and primitives use the legacy
        ``open_gripper`` / ``close_gripper`` driver methods instead.
        """
        return self.dof - 1 if self.dof >= 7 else None

    @model_validator(mode="after")
    def home_matches_dof(self) -> Self:
        """Validate sub-model lengths and home pose are consistent with dof."""
        if len(self.home_position) != self.dof:
            msg = f"home_position length ({len(self.home_position)}) must match dof ({self.dof})"
            raise ValueError(msg)
        if len(self.joint_limits) != self.dof:
            msg = f"joint_limits length ({len(self.joint_limits)}) must match dof ({self.dof})"
            raise ValueError(msg)
        if len(self.servos) != self.dof:
            msg = f"servos length ({len(self.servos)}) must match dof ({self.dof})"
            raise ValueError(msg)
        for idx, (lim, home) in enumerate(zip(self.joint_limits, self.home_position, strict=True)):
            if not (lim.min_rad <= home <= lim.max_rad):
                msg = (
                    f"home_position[{idx}] ({home}) outside joint_limits[{idx}] "
                    f"[{lim.min_rad}, {lim.max_rad}]"
                )
                raise ValueError(msg)
        if self.firmware.watchdog_timeout_s <= self.transport.keepalive_interval_s:
            msg = (
                f"firmware.watchdog_timeout_s ({self.firmware.watchdog_timeout_s}) "
                f"must exceed transport.keepalive_interval_s "
                f"({self.transport.keepalive_interval_s}) — otherwise the firmware "
                f"will latch e-stop while the driver is still keepalive-pinging."
            )
            raise ValueError(msg)
        return self


__all__ = [
    "ArmConfig",
    "ArmFirmwareConfig",
    "ArmServoConfig",
    "ArmTransportConfig",
    "JointLimits",
]
