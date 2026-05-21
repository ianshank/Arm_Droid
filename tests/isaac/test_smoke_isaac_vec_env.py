"""Isaac-gated smoke: build vec env with num_envs=2, run 5 steps, close (F1).

Auto-marked ``isaac`` + ``gpu`` by ``tests/isaac/conftest.py``.
Skipped unless ``ARMDROID_ISAAC_RUN=1`` AND ``isaaclab`` is importable.
"""

from __future__ import annotations

import torch

from armdroid.config.schema.sim_isaac import ArmSimIsaacConfig
from armdroid.config.schema.task import ArmTaskConfig
from armdroid.config.schema.training import ArmTrainingConfig
from armdroid.environments.isaac.reach_vec import SoArmReachIsaacVecEnv


def test_vec_env_lifecycle_num_envs_two(isaac_available: None) -> None:
    env = SoArmReachIsaacVecEnv(
        task_cfg=ArmTaskConfig(),
        training_cfg=ArmTrainingConfig(algorithm="rsl_rl_ppo"),
        sim_isaac_cfg=ArmSimIsaacConfig(num_envs=2),
    )
    try:
        obs, _ = env.reset(seed=0)
        assert env.num_envs == 2
        first_obs = next(iter(obs.values()))
        assert first_obs.shape[0] == 2

        action = torch.zeros(2, 6, dtype=torch.float32)
        for _ in range(5):
            obs, reward, term, trunc, _ = env.step(action)
            assert reward.shape == (2,)
            assert term.shape == (2,)
            assert trunc.shape == (2,)
    finally:
        env.close()


def test_vec_env_as_runner_env_returns_manager_based_rl_env(
    isaac_available: None,
) -> None:
    """The runner accessor returns the underlying isaaclab env directly."""
    env = SoArmReachIsaacVecEnv(
        task_cfg=ArmTaskConfig(),
        training_cfg=ArmTrainingConfig(algorithm="rsl_rl_ppo"),
        sim_isaac_cfg=ArmSimIsaacConfig(num_envs=2),
    )
    try:
        runner_env = env.as_runner_env()
        # Isaac Lab's vec env always exposes ``num_envs`` at the top level.
        assert getattr(runner_env, "num_envs", None) == 2
    finally:
        env.close()
