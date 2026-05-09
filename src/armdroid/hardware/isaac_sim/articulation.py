"""Isaac Lab ArticulationCfg builder for the SO-ARM101 (PR-B B.8).

Vendored from MuammerBay/isaac_so_arm101 @ e4624dea075b00a36dbc66bebd531d191c92e8cd
under BSD 3-Clause License (Copyright 2025, Muammer Bay (LycheeAI),
Louis Le Lay). Source file:
src/isaac_so_arm101/robots/trs_so100/so_arm100.py.

Modifications from upstream:
- Every numeric value (PD gains, init pose, solver iterations, fix_base,
  self_collision, root quat) is parametrised on
  :class:`ArmSimIsaacConfig` rather than hardcoded module constants.
  The numeric defaults in ``ArmSimIsaacConfig`` match upstream exactly,
  so behaviour is identical when ``sim_cfg`` is the default.
- Asset path resolved via ``armdroid.config.paths.resolve_asset_path``
  rather than ``Path(__file__).parent`` so the URDF is found
  regardless of CWD.
- Lazy isaaclab imports inside the function body so this module is
  importable on default installs (the ``[isaac]`` extra is required
  to *call* the function, not to import it).

Coverage-omit: this module is in ``[tool.coverage.run].omit`` because
its body imports isaaclab. Tests live under ``tests/isaac/`` and only
run with ``ARMDROID_ISAAC_RUN=1`` + a CUDA GPU.

THIRD_PARTY_NOTICES.md indexes the full attribution chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from armdroid.config.schema.arm import ArmConfig
    from armdroid.config.schema.sim_isaac import ArmSimIsaacConfig

from armdroid.config.paths import resolve_asset_path
from armdroid.logging.setup import get_logger

_log = get_logger(__name__)


def build_so_arm100_articulation_cfg(
    sim_cfg: ArmSimIsaacConfig,
    arm_cfg: ArmConfig,
) -> Any:
    """Build an Isaac Lab ``ArticulationCfg`` for the SO-ARM101.

    Args:
        sim_cfg: Isaac Sim configuration. Every numeric / string knob
            (PD gains, init pose, joint names, solver iterations) is
            sourced from this object — no hardcoded values.
        arm_cfg: Arm hardware configuration. Currently unused (DOF /
            joint limits / urdf_path live on ``sim_cfg``); reserved for
            future per-arm overrides.

    Returns:
        ``isaaclab.assets.ArticulationCfg`` ready to be passed to
        ``isaaclab.assets.Articulation`` or registered with a scene.

    Raises:
        ImportError: If the ``[isaac]`` extra is not installed
            (``isaaclab`` cannot be imported).
    """
    # Lazy imports — module top-level must NOT load isaaclab.
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    asset_path = resolve_asset_path(sim_cfg.urdf_fallback_path)
    _log.info(
        "isaac_sim_articulation_build",
        asset_path=str(asset_path),
        num_joints=len(sim_cfg.arm_joint_names) + 1,  # +1 for gripper
        fix_base=sim_cfg.fix_base,
    )

    # Per-joint stiffness / damping dicts — keyed on URDF joint names so
    # ImplicitActuatorCfg can resolve the regex against them. NB: the
    # upstream convention groups arm joints under one actuator + gripper
    # under another; we preserve that.
    arm_stiffness = {
        sim_cfg.arm_joint_names[0]: sim_cfg.arm_stiffness_shoulder_pan,
        sim_cfg.arm_joint_names[1]: sim_cfg.arm_stiffness_shoulder_lift,
        sim_cfg.arm_joint_names[2]: sim_cfg.arm_stiffness_elbow_flex,
        sim_cfg.arm_joint_names[3]: sim_cfg.arm_stiffness_wrist_flex,
        sim_cfg.arm_joint_names[4]: sim_cfg.arm_stiffness_wrist_roll,
    }
    arm_damping = {
        sim_cfg.arm_joint_names[0]: sim_cfg.arm_damping_shoulder_pan,
        sim_cfg.arm_joint_names[1]: sim_cfg.arm_damping_shoulder_lift,
        sim_cfg.arm_joint_names[2]: sim_cfg.arm_damping_elbow_flex,
        sim_cfg.arm_joint_names[3]: sim_cfg.arm_damping_wrist_flex,
        sim_cfg.arm_joint_names[4]: sim_cfg.arm_damping_wrist_roll,
    }

    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(asset_path),
            fix_base=sim_cfg.fix_base,
            replace_cylinders_with_capsules=True,  # Upstream MuammerBay default
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=sim_cfg.self_collisions,
                solver_position_iteration_count=sim_cfg.solver_position_iterations,
                solver_velocity_iteration_count=sim_cfg.solver_velocity_iterations,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(
                sim_cfg.init_root_pos_x,
                sim_cfg.init_root_pos_y,
                sim_cfg.init_root_pos_z,
            ),
            rot=sim_cfg.init_root_quat_wxyz,
            joint_pos={
                sim_cfg.arm_joint_names[0]: sim_cfg.init_shoulder_pan,
                sim_cfg.arm_joint_names[1]: sim_cfg.init_shoulder_lift,
                sim_cfg.arm_joint_names[2]: sim_cfg.init_elbow_flex,
                sim_cfg.arm_joint_names[3]: sim_cfg.init_wrist_flex,
                sim_cfg.arm_joint_names[4]: sim_cfg.init_wrist_roll,
                sim_cfg.gripper_joint_name: sim_cfg.init_gripper,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(sim_cfg.arm_joint_names),
                effort_limit_sim=sim_cfg.arm_effort_limit_sim,
                velocity_limit_sim=sim_cfg.arm_velocity_limit_sim,
                stiffness=arm_stiffness,
                damping=arm_damping,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[sim_cfg.gripper_joint_name],
                effort_limit_sim=sim_cfg.gripper_effort_limit_sim,
                velocity_limit_sim=sim_cfg.gripper_velocity_limit_sim,
                stiffness=sim_cfg.gripper_stiffness,
                damping=sim_cfg.gripper_damping,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


__all__ = ["build_so_arm100_articulation_cfg"]
