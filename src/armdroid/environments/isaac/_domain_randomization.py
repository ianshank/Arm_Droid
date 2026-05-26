"""Shared domain-randomization wiring for Isaac Lab reach environments.

Applies friction, mass, and actuator-gain randomization to an Isaac Lab
``ManagerBasedRLEnvCfg`` when ``ArmSimIsaacConfig.randomize_physics`` is
enabled.  Called by both the single-env and vec-env build paths so the
DR logic lives in exactly one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from armdroid.logging.setup import get_logger

if TYPE_CHECKING:
    from armdroid.config.schema.sim_isaac import ArmSimIsaacConfig

_log = get_logger(__name__)


def apply_domain_randomization(
    env_cfg: Any,
    sim_cfg: ArmSimIsaacConfig,
) -> None:
    """Mutate *env_cfg* in-place to add physics randomization events.

    This is a no-op when ``sim_cfg.randomize_physics`` is ``False``.

    Args:
        env_cfg: A parsed Isaac Lab ``ManagerBasedRLEnvCfg`` (or subclass).
        sim_cfg: The armdroid Isaac Sim configuration carrying the
            randomization ranges.
    """
    if not sim_cfg.randomize_physics:
        return

    _log.info(
        "domain_randomization_enabled",
        friction=sim_cfg.friction_range_sim,
        mass=sim_cfg.mass_scale_range,
        stiffness_noise=sim_cfg.stiffness_noise_pct,
        damping_noise=sim_cfg.damping_noise_pct,
    )

    try:
        import isaaclab.envs.mdp as mdp
        from isaaclab.managers import EventTermCfg

        env_cfg.events.randomize_friction = EventTermCfg(
            func=mdp.randomize_rigid_body_friction,
            mode="reset",
            params={
                "asset_cfg": env_cfg.scene.robot,
                "friction_range": sim_cfg.friction_range_sim,
            },
        )
        env_cfg.events.randomize_mass = EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": env_cfg.scene.robot,
                "mass_distribution_params": sim_cfg.mass_scale_range,
                "operation": "scale",
            },
        )

        stiff_min = 1.0 - sim_cfg.stiffness_noise_pct / 100.0
        stiff_max = 1.0 + sim_cfg.stiffness_noise_pct / 100.0
        damp_min = 1.0 - sim_cfg.damping_noise_pct / 100.0
        damp_max = 1.0 + sim_cfg.damping_noise_pct / 100.0

        env_cfg.events.randomize_gains = EventTermCfg(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": env_cfg.scene.robot,
                "stiffness_distribution_params": (stiff_min, stiff_max),
                "damping_distribution_params": (damp_min, damp_max),
                "operation": "scale",
            },
        )
    except (ImportError, AttributeError) as exc:
        _log.warning("domain_randomization_mdp_missing", error=str(exc))


__all__ = ["apply_domain_randomization"]
