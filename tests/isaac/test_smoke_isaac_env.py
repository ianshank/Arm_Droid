"""SoArmReachIsaacEnv smoke test (PR-B B.17).

Runs locally only with ``ARMDROID_ISAAC_RUN=1 pytest tests/isaac``.
"""

from __future__ import annotations

import numpy as np


def test_isaac_env_reset_step_close(isaac_available: None) -> None:
    """Reset, step five times, close — full env lifecycle without errors."""
    from armdroid.config.schema import ArmSettings, ArmTaskConfig
    from armdroid.environments.isaac import SoArmReachIsaacEnv

    cfg = ArmSettings()
    env = SoArmReachIsaacEnv(
        ArmTaskConfig(task_type="so_arm_reach_isaac"),
        cfg.arm_training,
        dof=cfg.arm.dof,
        sim_isaac_cfg=cfg.arm_sim_isaac,
    )
    obs, info = env.reset(seed=0)
    assert isinstance(obs, dict)
    assert isinstance(info, dict)

    zero_action = np.zeros(cfg.arm.dof, dtype=np.float64)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    env.close()
