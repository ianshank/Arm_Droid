"""Root configuration schema for armdroid — single source of truth.

All values read from YAML config files or environment variables.
Nothing hardcoded elsewhere. New fields MUST have defaults (backwards
compatibility guarantee).

Ported from mousedroid arm config block at schema.py:1754-1980 with:
- ``yolo_backend`` literal collapsed to ``"ultralytics"`` only (Hailo
  is Jetson hardware; armdroid runs on a desktop with a CUDA GPU).
- Legacy field aliases (rover's ``arm_hardware``, ``arm_simulation``, …)
  dropped — armdroid v0.1.0 has no historical YAML to support.
- Top-level Settings root replaced with :class:`ArmSettings` using
  ``ARMDROID_`` env prefix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from armdroid.config.logging import LoggingConfig

# ---------------------------------------------------------------------------
# LLM replanner sub-config (used by ArmPlanningConfig)
# ---------------------------------------------------------------------------


class LLMReplannerConfig(BaseModel):
    """Configuration for the LLM-backed arm replanner.

    Disabled by default; when enabled, ``backend`` selects the concrete
    implementation. ``model``, ``max_tokens``, ``temperature`` and the
    request envelope come from this config so no values are hardcoded
    in the backend modules.
    """

    enabled: bool = Field(
        False,
        description="Enable LLM-backed replanning",
    )
    backend: Literal["null", "llama", "anthropic"] = Field(
        "null",
        description="Replanner backend selection",
    )
    model: str = Field(
        "claude-sonnet-4-6",
        description="Model identifier passed to the backend",
    )
    max_tokens: int = Field(
        1024,
        gt=0,
        description="Per-request max tokens",
    )
    temperature: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    system_prompt: str = Field(
        "",
        description="System prompt passed to the backend",
    )
    api_key_env_var: str = Field(
        "ANTHROPIC_API_KEY",
        description="Env var holding the API key (Anthropic backend only)",
    )
    request_timeout_s: float = Field(
        30.0,
        gt=0,
        description="Per-request timeout (s)",
    )
    max_retries: int = Field(
        3,
        ge=0,
        description="Max exponential-backoff retries on transient errors",
    )


# ---------------------------------------------------------------------------
# Robot arm sub-configs
# ---------------------------------------------------------------------------


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
        Path("urdf/so_arm100.urdf"),
        description="Path to robot arm URDF file",
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
        "pose back to home_position. Used by primitives.home() and on "
        "MockArmDriver.connect().",
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
        description="Duration over which the firmware interpolates a grasp "
        "primitive's approach phase. Gripper close is instantaneous.",
    )
    place_duration_s: float = Field(
        default=1.0,
        gt=0.0,
        description="Duration over which the firmware interpolates a place "
        "primitive's deposit phase. Gripper open is instantaneous.",
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
            base_pins = [13, 14, 27, 26, 25, 33, 32, 15, 4, 5, 18, 19]
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


class ArmSimConfig(BaseModel):
    """MuJoCo simulation configuration for robot arm training."""

    scene_path: Path = Field(
        Path("sim/tower_of_hanoi.xml"),
        description="MuJoCo scene XML path",
    )
    timestep_s: float = Field(0.002, gt=0, description="Physics timestep (s)")
    n_substeps: int = Field(5, gt=0, description="Physics substeps per step")
    render_width: int = Field(640, gt=0, description="Render width (px)")
    render_height: int = Field(480, gt=0, description="Render height (px)")
    domain_randomization: bool = Field(True, description="Enable domain randomization")
    mass_range_pct: float = Field(20.0, ge=0, le=100, description="Mass variation range (%)")
    friction_range: float = Field(0.3, ge=0, description="Friction coefficient variation")
    position_noise_m: float = Field(0.005, ge=0, description="Object position noise (m)")
    lighting_variation: float = Field(0.2, ge=0, le=1, description="Lighting intensity variation")
    camera_pose_noise_deg: float = Field(10.0, ge=0, description="Camera pose noise (degrees)")


class ArmPerceptionConfig(BaseModel):
    """Perception stack configuration for robot arm platform.

    Note: ``yolo_backend`` is fixed to ``"ultralytics"`` in armdroid (the
    Hailo accelerator path was Jetson-specific and has been removed).
    The field is preserved for forward compatibility if other backends
    (e.g. TensorRT) are added later.
    """

    depth_camera_type: Literal["realsense_d435i", "oak_d", "zed2i", "mock"] = Field(
        "realsense_d435i",
        description="Depth camera hardware type",
    )
    yolo_model_path: Path = Field(
        Path("models/yolo11_disk_detector.pt"),
        description="YOLO model weights path",
    )
    yolo_confidence_threshold: float = Field(
        0.5, gt=0, le=1, description="YOLO detection confidence threshold"
    )
    yolo_nms_iou_threshold: float = Field(
        0.45,
        gt=0,
        le=1,
        description="YOLO NMS IoU threshold for non-maximum suppression",
    )
    yolo_backend: Literal["ultralytics"] = Field(
        "ultralytics",
        description="YOLO inference backend (ultralytics on CUDA)",
    )
    pose_estimator: Literal["pnp", "learned"] = Field(
        "pnp",
        description="Pose estimation method",
    )
    pose_tolerance_m: float = Field(0.005, gt=0, description="Pose estimation tolerance (m)")
    detection_fps: float = Field(30.0, gt=0, description="Detection rate (Hz)")
    depth_min_m: float = Field(0.01, gt=0, description="Minimum valid depth (m)")
    depth_max_m: float = Field(10.0, gt=0, description="Maximum valid depth (m)")
    depth_hole_threshold_m: float = Field(
        0.02, gt=0, description="Depth below which pixels are treated as holes (m)"
    )
    depth_filter_kernel_size: int = Field(
        3, gt=0, description="Median filter kernel size for depth noise reduction"
    )
    fallback_depth_m: float = Field(
        0.3, gt=0, description="Fallback depth when centre pixel is invalid (m)"
    )
    invalid_depth_threshold_m: float = Field(
        0.01, ge=0, description="Depth values below this are considered invalid (m)"
    )
    white_brightness_threshold: float = Field(
        200.0, ge=0, le=255, description="Brightness above which garment is classified white"
    )
    white_saturation_threshold: float = Field(
        0.15, ge=0, le=1, description="Saturation below which bright garment is white"
    )
    dark_brightness_threshold: float = Field(
        80.0, ge=0, le=255, description="Brightness below which garment is classified dark"
    )
    default_focal_length: float = Field(500.0, gt=0, description="Default camera focal length (px)")
    default_principal_x: float = Field(320.0, gt=0, description="Default principal point X (px)")
    default_principal_y: float = Field(240.0, gt=0, description="Default principal point Y (px)")


class ArmPlanningConfig(BaseModel):
    """Symbolic planning configuration for robot arm tasks."""

    pddl_domain_path: Path = Field(
        Path("planning/pddl/hanoi_domain.pddl"),
        description="PDDL domain file path",
    )
    planner_backend: Literal["pyperplan", "fast_downward"] = Field(
        "pyperplan",
        description="PDDL solver backend",
    )
    llm_replanner_enabled: bool = Field(
        False,
        description="Enable LLM-based adaptive replanning on execution failure",
    )
    max_replan_attempts: int = Field(3, gt=0, description="Max replanning attempts before abort")
    planning_timeout_s: float = Field(5.0, gt=0, description="Maximum planning time (s)")
    llm_replanner: LLMReplannerConfig | None = Field(
        None,
        description="LLM-backed replanner config (None=use legacy symbolic fallback)",
    )


class ArmTrainingConfig(BaseModel):
    """RL training configuration for robot arm policies."""

    algorithm: Literal["sac", "ppo", "sac_her"] = Field(
        "sac_her",
        description="RL algorithm (SAC, PPO, or SAC+HER)",
    )
    learning_rate: float = Field(3e-4, gt=0, description="Policy learning rate")
    batch_size: int = Field(256, gt=0, description="Training batch size")
    buffer_size: int = Field(1_000_000, gt=0, description="Replay buffer capacity")
    gamma: float = Field(0.99, gt=0, le=1, description="Discount factor")
    tau: float = Field(0.005, gt=0, le=1, description="Soft target update coefficient")
    total_timesteps: int = Field(1_000_000, gt=0, description="Total training timesteps")
    eval_frequency: int = Field(10_000, gt=0, description="Evaluation frequency (steps)")
    checkpoint_frequency: int = Field(50_000, gt=0, description="Checkpoint save frequency (steps)")
    n_eval_episodes: int = Field(20, gt=0, description="Episodes per evaluation")
    video_frequency: int = Field(50_000, gt=0, description="Video rollout frequency (steps)")
    her_n_sampled_goal: int = Field(4, gt=0, description="HER goal relabeling ratio")
    her_goal_selection: Literal["future", "final", "episode"] = Field(
        "future",
        description="HER goal selection strategy",
    )
    reward_grasp: float = Field(0.1, description="Reward for successful grasp")
    reward_place: float = Field(0.2, description="Reward for correct placement")
    reward_complete: float = Field(1.0, description="Reward for task completion")
    penalty_collision: float = Field(-0.5, description="Penalty for collision")
    penalty_wrong_disk: float = Field(-0.1, description="Penalty for grasping wrong disk")
    seed: int = Field(42, ge=0, description="Random seed for reproducibility")
    weights_dir: str = Field("weights/arm", description="Checkpoint output directory")
    action_delta_min: float = Field(-0.1, description="Minimum action delta per step (rad)")
    action_delta_max: float = Field(0.1, gt=0, description="Maximum action delta per step (rad)")
    distance_penalty_coeff: float = Field(
        0.01, ge=0, description="Dense distance-based reward penalty coefficient"
    )


class ArmCurriculumConfig(BaseModel):
    """Curriculum learning configuration for progressive task difficulty."""

    enabled: bool = Field(True, description="Enable curriculum learning")
    stages: list[int] = Field(
        default_factory=lambda: [1, 2, 3, 5],
        description="Curriculum stages (number of disks per stage)",
    )
    promotion_threshold: float = Field(
        0.8, gt=0, le=1, description="Success rate threshold to advance stage"
    )
    promotion_eval_episodes: int = Field(
        50, gt=0, description="Episodes to evaluate before stage promotion"
    )
    warm_start: bool = Field(True, description="Warm-start from previous stage weights")


class ArmTaskConfig(BaseModel):
    """Task-specific configuration for robot arm manipulation tasks."""

    task_type: Literal["tower_of_hanoi", "laundry_sorting", "pick_place"] = Field(
        "tower_of_hanoi",
        description="Manipulation task type",
    )
    num_disks: int = Field(3, gt=0, le=10, description="Number of disks (Tower of Hanoi)")
    num_pegs: int = Field(3, gt=1, le=5, description="Number of pegs (Tower of Hanoi)")
    peg_positions: list[list[float]] = Field(
        default_factory=lambda: [[0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [0.4, 0.0, 0.0]],
        description="Peg XYZ positions (m) — length must match num_pegs",
    )
    num_baskets: int = Field(3, gt=0, le=5, description="Number of sorting baskets (laundry)")
    basket_positions: list[list[float]] = Field(
        default_factory=lambda: [[0.2, -0.2, 0.0], [0.3, -0.2, 0.0], [0.4, -0.2, 0.0]],
        description="Basket XYZ positions (m) — length must match num_baskets",
    )
    max_episode_steps: int = Field(500, gt=0, description="Max steps per episode")
    num_garments: int = Field(5, gt=0, description="Number of garments per episode (laundry)")

    @model_validator(mode="after")
    def positions_match_count(self) -> Self:
        """Validate position list lengths match counts."""
        if len(self.peg_positions) != self.num_pegs:
            msg = (
                f"peg_positions length ({len(self.peg_positions)})"
                f" must match num_pegs ({self.num_pegs})"
            )
            raise ValueError(msg)
        if len(self.basket_positions) != self.num_baskets:
            msg = (
                f"basket_positions length ({len(self.basket_positions)})"
                f" must match num_baskets ({self.num_baskets})"
            )
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


class ArmSettings(BaseSettings):
    """Root settings for armdroid.

    All sub-configs default to constructed instances so ``ArmSettings()``
    produces a complete usable configuration without any YAML overlay.
    YAML files override individual fields; ``ARMDROID_*`` env vars
    override everything.
    """

    model_config = SettingsConfigDict(
        env_prefix="ARMDROID_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    arm: ArmConfig = Field(default_factory=ArmConfig)
    arm_sim: ArmSimConfig = Field(default_factory=ArmSimConfig)
    arm_perception: ArmPerceptionConfig = Field(default_factory=ArmPerceptionConfig)
    arm_planning: ArmPlanningConfig = Field(default_factory=ArmPlanningConfig)
    arm_training: ArmTrainingConfig = Field(default_factory=ArmTrainingConfig)
    arm_curriculum: ArmCurriculumConfig = Field(default_factory=ArmCurriculumConfig)
    arm_task: ArmTaskConfig = Field(default_factory=ArmTaskConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    mock_hardware: bool = Field(
        default=False,
        description="Use mock hardware drivers instead of real serial/USB devices.",
    )


def load_settings(*overlay_paths: Path, config_dir: Path | None = None) -> ArmSettings:
    """Convenience wrapper: load ArmSettings from YAML overlays + env vars.

    Args:
        overlay_paths: YAML files to merge on top of ``<config_dir>/default.yaml``.
        config_dir: Directory containing ``default.yaml``. Defaults to repo ``config/``.

    Returns:
        Fully resolved :class:`ArmSettings`.
    """
    from armdroid.config.loader import load_settings as _generic_load

    return _generic_load(
        *overlay_paths,
        settings_class=ArmSettings,
        config_dir=config_dir,
    )
