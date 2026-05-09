"""Isaac Sim 5.1 / Isaac Lab 2.3 configuration.

Every numeric servo PD gain, init pose, env id, joint name, and gripper
conversion knob is a Pydantic field with bounds — zero hardcoded values
in the IsaacSimDriver / SoArmReachIsaacEnv consumers. Defaults are
placeholders pinned at vendor time to MuammerBay/isaac_so_arm101 upstream
values; the upstream commit SHA is captured in
``assets/so_arm/so101/ATTRIBUTION.md`` and ``THIRD_PARTY_NOTICES.md``.

PR-B B.2 — see the implementation plan for the full field rationale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from armdroid.config.paths import SO101_URDF_REL


class ArmSimIsaacConfig(BaseModel):
    """Isaac Sim 5.1 / Isaac Lab 2.3 backend configuration.

    Consumed by ``armdroid.hardware.isaac_sim.{driver,articulation}``
    and ``armdroid.environments.isaac.reach``. Every numeric / string
    knob the Isaac code-path needs lives here; the consumers do not
    hardcode any values.

    Field naming convention (matches existing schemas — `arm.py`,
    `sim.py`): snake_case, unit-suffixed (``_s`` for seconds, ``_m`` for
    metres, ``_pct`` for percent, ``_rad`` for radians,
    ``_sim`` for sim-only fields that may differ from the URDF/datasheet
    physical limits, ``_wxyz`` for quat order).
    """

    # ------------------------------------------------------------------ #
    # App launch
    # ------------------------------------------------------------------ #

    headless: bool = Field(
        default=True,
        description="Run Kit in headless mode (no viewport window). False enables GUI debugging.",
    )
    enable_cameras: bool = Field(
        default=False,
        description=(
            "Enable camera sensors during sim. Slow; enable only for perception experiments."
        ),
    )
    enable_logging: bool = Field(
        default=False,
        description="Verbose Kit / Isaac logging. Defaults off to keep the CLI quiet.",
    )

    # ------------------------------------------------------------------ #
    # Asset paths — defaults match the vendored SO-ARM101 layout
    # ------------------------------------------------------------------ #

    usd_path: Path = Field(
        default_factory=lambda: SO101_URDF_REL.parent / "usd" / "so101.usd",
        description=(
            "USD output path (build artefact, gitignored). Auto-converted "
            "from urdf_fallback_path on first AppLauncher boot."
        ),
    )
    urdf_fallback_path: Path = Field(
        default_factory=lambda: SO101_URDF_REL,
        description=(
            "URDF used by isaaclab.sim.UrdfFileCfg when usd_path is absent. "
            "Defaults to the vendored SO-ARM101 URDF."
        ),
    )

    # ------------------------------------------------------------------ #
    # Physics
    # ------------------------------------------------------------------ #

    physics_dt_s: float = Field(
        default=0.005,
        gt=0.0,
        le=0.05,
        description="Physics step (seconds). Inverse of physics rate; 0.005 = 200 Hz.",
    )
    decimation: int = Field(
        default=10,
        ge=1,
        le=200,
        description="Control-step decimation: control_dt = physics_dt_s * decimation.",
    )
    num_envs: int = Field(
        default=1,
        ge=1,
        le=16384,
        description=(
            "Number of parallel environments. Protocol path (IsaacSimDriver / "
            "SoArmReachIsaacEnv) requires num_envs == 1; vectorised training "
            "bypasses the protocol via Isaac Lab's runner directly."
        ),
    )
    fix_base: bool = Field(
        default=True,
        description="Fix the articulation root to world (table-mount).",
    )
    self_collisions: bool = Field(
        default=True,
        description="Enable per-link self-collision checking.",
    )
    solver_position_iterations: int = Field(
        default=8,
        ge=1,
        le=64,
        description="PhysX position solver iterations per substep.",
    )
    solver_velocity_iterations: int = Field(
        default=0,
        ge=0,
        le=32,
        description="PhysX velocity solver iterations per substep (0 = position-only).",
    )

    # ------------------------------------------------------------------ #
    # Init root pose (peer-review hole-finder: was hardcoded in B.8 v1)
    # ------------------------------------------------------------------ #

    init_root_pos_x: float = Field(default=0.0, description="Initial root x position (m).")
    init_root_pos_y: float = Field(default=0.0, description="Initial root y position (m).")
    init_root_pos_z: float = Field(default=0.5, description="Initial root z position (m).")
    init_root_quat_wxyz: tuple[float, float, float, float] = Field(
        default=(0.7071, 0.0, 0.0, 0.7071),
        description="Initial root quaternion (w, x, y, z). Default rotates the arm 90° about Z.",
    )

    # ------------------------------------------------------------------ #
    # Init pose (radians) — per-joint
    # ------------------------------------------------------------------ #

    init_shoulder_pan: float = Field(default=0.0, description="shoulder_pan init position (rad).")
    init_shoulder_lift: float = Field(
        default=1.57, description="shoulder_lift init position (rad)."
    )
    init_elbow_flex: float = Field(default=-1.57, description="elbow_flex init position (rad).")
    init_wrist_flex: float = Field(default=1.0, description="wrist_flex init position (rad).")
    init_wrist_roll: float = Field(default=-1.57, description="wrist_roll init position (rad).")
    init_gripper: float = Field(
        default=0.0, description="gripper init position (rad, URDF convention)."
    )

    # ------------------------------------------------------------------ #
    # Arm actuator group (5 revolute joints)
    # ------------------------------------------------------------------ #

    arm_effort_limit_sim: float = Field(
        default=1.9, gt=0.0, description="Per-joint effort limit (N·m, sim-tuned)."
    )
    arm_velocity_limit_sim: float = Field(
        default=1.5, gt=0.0, description="Per-joint velocity limit (rad/s, sim-tuned)."
    )

    arm_stiffness_shoulder_pan: float = Field(
        default=200.0, ge=0.0, description="PD stiffness K_p."
    )
    arm_stiffness_shoulder_lift: float = Field(
        default=170.0, ge=0.0, description="PD stiffness K_p."
    )
    arm_stiffness_elbow_flex: float = Field(default=120.0, ge=0.0, description="PD stiffness K_p.")
    arm_stiffness_wrist_flex: float = Field(default=80.0, ge=0.0, description="PD stiffness K_p.")
    arm_stiffness_wrist_roll: float = Field(default=50.0, ge=0.0, description="PD stiffness K_p.")

    arm_damping_shoulder_pan: float = Field(default=80.0, ge=0.0, description="PD damping K_d.")
    arm_damping_shoulder_lift: float = Field(default=65.0, ge=0.0, description="PD damping K_d.")
    arm_damping_elbow_flex: float = Field(default=45.0, ge=0.0, description="PD damping K_d.")
    arm_damping_wrist_flex: float = Field(default=30.0, ge=0.0, description="PD damping K_d.")
    arm_damping_wrist_roll: float = Field(default=20.0, ge=0.0, description="PD damping K_d.")

    # ------------------------------------------------------------------ #
    # Gripper actuator
    # ------------------------------------------------------------------ #

    gripper_stiffness: float = Field(default=60.0, ge=0.0, description="Gripper PD stiffness K_p.")
    gripper_damping: float = Field(default=20.0, ge=0.0, description="Gripper PD damping K_d.")
    gripper_effort_limit_sim: float = Field(
        default=2.5, gt=0.0, description="Gripper effort limit (N, sim-tuned)."
    )
    gripper_velocity_limit_sim: float = Field(
        default=1.5, gt=0.0, description="Gripper velocity limit (m/s or rad/s, sim-tuned)."
    )

    # ------------------------------------------------------------------ #
    # Gripper unit conversion (closes R5 + R7) — single source of truth
    # for the rescale-and-invert at the URDF↔armdroid boundary
    # ------------------------------------------------------------------ #

    gripper_joint_radians_open: float = Field(
        default=1.74533,
        description=(
            "Gripper joint position (rad) corresponding to fully open jaw "
            "in the URDF convention. URDF maps positive radians to open."
        ),
    )
    gripper_joint_radians_closed: float = Field(
        default=0.0,
        description="Gripper joint position (rad) corresponding to fully closed jaw.",
    )
    gripper_normalised_open: float = Field(
        default=0.0,
        description=(
            "armdroid normalised value for fully-open gripper. armdroid convention: 0=open."
        ),
    )
    gripper_normalised_closed: float = Field(
        default=1.0,
        description=(
            "armdroid normalised value for fully-closed gripper. armdroid convention: 1=closed."
        ),
    )

    # ------------------------------------------------------------------ #
    # Gripper joint position (peer-review hole-finder fix)
    # ------------------------------------------------------------------ #

    gripper_joint_index: int = Field(
        default=5,
        ge=0,
        le=11,
        description=(
            "Index of the gripper in the joint vector passed to "
            "send_joint_positions(). Default 5 = last joint of a 6-DoF arm."
        ),
    )

    # ------------------------------------------------------------------ #
    # Gym env id
    # ------------------------------------------------------------------ #

    reach_env_id: str = Field(
        default="Isaac-SO-ARM100-Reach-v0",
        description="gymnasium env id used by gym.make() in SoArmReachIsaacEnv.",
    )
    reach_play_env_id: str = Field(
        default="Isaac-SO-ARM100-Reach-Play-v0",
        description="gymnasium env id for evaluation rollouts (no domain randomisation).",
    )

    # ------------------------------------------------------------------ #
    # Joint names — used by ImplicitActuatorCfg.joint_names_expr regexes
    # ------------------------------------------------------------------ #

    arm_joint_names: tuple[str, ...] = Field(
        default=("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"),
        description="Names of the 5 arm joints in URDF order (matches MuammerBay/isaac_so_arm101).",
    )
    gripper_joint_name: str = Field(
        default="gripper",
        description="Name of the gripper joint in the URDF.",
    )

    # ------------------------------------------------------------------ #
    # Schedule type — used by RSL-RL PPO learning rate schedule
    # ------------------------------------------------------------------ #
    # (NB: this field is here rather than in RslRlPpoConfig because the
    # decimation schedule depends on physics_dt_s which is ours, not
    # rsl-rl's.)

    schedule_kind: Literal["adaptive", "fixed"] = Field(
        default="adaptive",
        description="Default LR schedule kind. RslRlPpoConfig may override.",
    )


def _default_sim_cfg() -> ArmSimIsaacConfig:
    """Return a fresh default Isaac Sim config.

    Used as a fallback by ``IsaacSimDriver`` and ``SoArmReachIsaacEnv``
    when no explicit ``sim_isaac_cfg`` is passed. Construct fresh each
    call (not a module-level constant) so YAML / env overrides set
    later in the process still apply via Pydantic's BaseSettings
    re-read.
    """
    return ArmSimIsaacConfig()


__all__ = ["ArmSimIsaacConfig", "_default_sim_cfg"]
