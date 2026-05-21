"""Unit tests for SoArmReachIsaacVecEnv with a mocked ManagerBasedRLEnv (F1).

All ``isaaclab``-touching paths are stubbed via ``_build_isaac_env``
monkeypatching, so these tests run on the default install.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from armdroid.config.schema.sim_isaac import ArmSimIsaacConfig
from armdroid.config.schema.task import ArmTaskConfig
from armdroid.config.schema.training import ArmTrainingConfig
from armdroid.domain.protocols import VecArmEnvironmentProtocol
from tests.helpers.fake_isaac_env import make_fake_isaac_env


@pytest.fixture(autouse=True)
def _reset_app_state() -> Any:
    """Each test starts with a clean AppLauncher singleton flag."""
    from armdroid.hardware.isaac_sim import _app_state

    _app_state.reset_for_tests()
    yield
    _app_state.reset_for_tests()


def test_vec_env_satisfies_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    from armdroid.environments.isaac import reach_vec

    monkeypatch.setattr(
        reach_vec, "_build_isaac_env", lambda *_a, **_kw: make_fake_isaac_env(num_envs=4),
    )
    env = reach_vec.SoArmReachIsaacVecEnv(
        task_cfg=ArmTaskConfig(),
        training_cfg=ArmTrainingConfig(),
        sim_isaac_cfg=ArmSimIsaacConfig(num_envs=4),
    )
    assert isinstance(env, VecArmEnvironmentProtocol)
    assert env.num_envs == 4


def test_vec_env_reset_step_close(monkeypatch: pytest.MonkeyPatch) -> None:
    from armdroid.environments.isaac import reach_vec

    fake = make_fake_isaac_env(num_envs=4)
    monkeypatch.setattr(reach_vec, "_build_isaac_env", lambda *_a, **_kw: fake)
    env = reach_vec.SoArmReachIsaacVecEnv(
        task_cfg=ArmTaskConfig(),
        training_cfg=ArmTrainingConfig(),
        sim_isaac_cfg=ArmSimIsaacConfig(num_envs=4),
    )

    obs, info = env.reset(seed=42)
    assert obs["observation"].shape == (4, 6)
    assert isinstance(info, dict)

    action = torch.zeros(4, 6, dtype=torch.float32)
    next_obs, reward, term, trunc, info2 = env.step(action)
    assert reward.shape == (4,)
    assert term.shape == (4,)
    assert trunc.shape == (4,)

    runner_env = env.as_runner_env()
    assert runner_env is fake

    env.close()
    fake.close.assert_called_once()


def test_vec_env_accepts_num_envs_one_as_degenerate_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``num_envs == 1`` is permitted on the vec env (degenerate case).

    The factory still routes ``num_envs == 1`` to the single-env path;
    this test guards the vec env contract itself so callers may adopt
    the vec protocol unconditionally.
    """
    from armdroid.environments.isaac import reach_vec

    fake = make_fake_isaac_env(num_envs=1)
    monkeypatch.setattr(reach_vec, "_build_isaac_env", lambda *_a, **_kw: fake)
    env = reach_vec.SoArmReachIsaacVecEnv(
        task_cfg=ArmTaskConfig(),
        training_cfg=ArmTrainingConfig(),
        sim_isaac_cfg=ArmSimIsaacConfig(num_envs=1),
    )
    assert env.num_envs == 1
    env.reset(seed=0)


def test_vec_env_default_sim_cfg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting ``sim_isaac_cfg`` falls back to the default factory."""
    from armdroid.environments.isaac import reach_vec

    monkeypatch.setattr(
        reach_vec, "_build_isaac_env", lambda *_a, **_kw: make_fake_isaac_env(num_envs=1),
    )
    env = reach_vec.SoArmReachIsaacVecEnv(
        task_cfg=ArmTaskConfig(),
        training_cfg=ArmTrainingConfig(),
    )
    assert env.num_envs >= 1


def test_vec_env_as_runner_env_triggers_lazy_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``as_runner_env`` must build the underlying env if not already built."""
    from armdroid.environments.isaac import reach_vec

    fake = make_fake_isaac_env(num_envs=2)
    monkeypatch.setattr(reach_vec, "_build_isaac_env", lambda *_a, **_kw: fake)
    env = reach_vec.SoArmReachIsaacVecEnv(
        task_cfg=ArmTaskConfig(),
        training_cfg=ArmTrainingConfig(),
        sim_isaac_cfg=ArmSimIsaacConfig(num_envs=2),
    )
    # No reset / step yet
    runner_env = env.as_runner_env()
    assert runner_env is fake
