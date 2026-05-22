"""Hypothesis property tests for VecArmEnvironmentProtocol (F1).

Runs unconditionally on every CI shard: ``_build_isaac_env`` is
monkeypatched to a torch-returning ``MagicMock`` so isaaclab is never
imported. The Hypothesis loop explores ``num_envs`` in ``[1, 8]``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from armdroid.config.schema.sim_isaac import ArmSimIsaacConfig
from armdroid.config.schema.task import ArmTaskConfig
from armdroid.config.schema.training import ArmTrainingConfig
from tests.helpers.fake_isaac_env import make_fake_isaac_env
from tests.property._vec_invariants import (
    assert_reset_shape,
    assert_step_shapes,
)


@pytest.fixture(autouse=True)
def _reset_app_state() -> Any:
    from armdroid.hardware.isaac_sim import _app_state

    _app_state.reset_for_tests()
    yield
    _app_state.reset_for_tests()


@given(num_envs=st.integers(min_value=1, max_value=8))
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_vec_env_shapes_consistent_with_num_envs(
    num_envs: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reset + step shapes must match the configured num_envs (including 1)."""
    from armdroid.environments.isaac import reach_vec

    fake = make_fake_isaac_env(num_envs=num_envs)
    monkeypatch.setattr(reach_vec, "_build_isaac_env", lambda *_a, **_kw: fake)
    env = reach_vec.SoArmReachIsaacVecEnv(
        task_cfg=ArmTaskConfig(),
        training_cfg=ArmTrainingConfig(),
        sim_isaac_cfg=ArmSimIsaacConfig(num_envs=num_envs),
    )
    assert env.num_envs == num_envs

    obs, info = env.reset(seed=0)
    assert_reset_shape(obs, info, num_envs=num_envs)

    action = torch.zeros(num_envs, 6, dtype=torch.float32)
    obs2, reward, term, trunc, info2 = env.step(action)
    assert_step_shapes(obs2, reward, term, trunc, info2, num_envs=num_envs)
