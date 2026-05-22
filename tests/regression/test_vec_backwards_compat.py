"""Regression: single-env path is byte-identical post-F1.

These tests pin the backwards-compat guarantees of the F1 vec env
landing: the existing ``_TensorAdapter`` seam still raises on
``num_envs > 1``, the single-env adapter accepts ``num_envs == 1``,
and ``RslRlPpoAgent.build`` still reach-throughs to ``env._isaac_env``
(no deprecation warning added; the attribute is private).
"""

from __future__ import annotations

import inspect

import pytest

from armdroid.environments._tensor_adapter import _TensorAdapter


def test_tensor_adapter_still_rejects_num_envs_gt_one() -> None:
    """The single-env adapter remains the seam between paths."""
    with pytest.raises(RuntimeError, match="num_envs="):
        _TensorAdapter(num_envs=2)


def test_single_env_path_still_accepts_num_envs_one() -> None:
    """num_envs == 1 default path remains supported by the adapter."""
    adapter = _TensorAdapter(num_envs=1)
    assert adapter is not None


def test_rsl_rl_ppo_build_still_uses_isaac_env_reach_through() -> None:
    """The legacy ``build()`` reach-through to ``env._isaac_env`` is preserved.

    Pins the backwards-compat guarantee: existing callers of
    ``RslRlPpoAgent.build(single_env)`` continue to work exactly as
    they did before F1. The body must still contain the
    ``_isaac_env`` access for the single-env path.
    """
    import armdroid.control.rsl_rl_agent as agent_mod

    source = inspect.getsource(agent_mod.RslRlPpoAgent.build)
    assert "_isaac_env" in source, (
        "Single-env build() must still reach through env._isaac_env for "
        "backwards-compat; F1 explicitly does not modify this path."
    )


def test_iterations_for_helper_extraction_preserves_behaviour() -> None:
    """``_iterations_for`` returns the same values as the prior inline code.

    Pure-extraction regression: the old inline computation
    ``int(total_timesteps // num_steps_per_env) if total_timesteps else
    num_iterations`` is now ``self._iterations_for(total_timesteps)``.
    """
    from armdroid.config.schema.training import ArmTrainingConfig, RslRlPpoConfig
    from armdroid.control.rsl_rl_agent import RslRlPpoAgent

    agent = RslRlPpoAgent(
        ppo_cfg=RslRlPpoConfig(),
        training_cfg=ArmTrainingConfig(algorithm="rsl_rl_ppo"),
        device="cpu",
    )
    # None path: returns the configured iteration count.
    assert agent._iterations_for(None) == agent._ppo_cfg.num_iterations
    # Explicit total_timesteps path: floor division by num_steps_per_env.
    expected = int(50_000 // agent._ppo_cfg.num_steps_per_env)
    assert agent._iterations_for(50_000) == expected


def test_iterations_for_floor_divides_when_not_evenly_divisible() -> None:
    """Non-evenly-divisible total_timesteps must floor-divide (no rounding up).

    Pins the truncation behaviour: a caller passing 1001 with
    num_steps_per_env=24 gets ``41`` iterations (24*41 = 984 < 1001),
    not 42 (24*42 = 1008 > 1001). Mirrors RSL-RL's iteration semantics.
    """
    from armdroid.config.schema.training import ArmTrainingConfig, RslRlPpoConfig
    from armdroid.control.rsl_rl_agent import RslRlPpoAgent

    agent = RslRlPpoAgent(
        ppo_cfg=RslRlPpoConfig(num_steps_per_env=24),
        training_cfg=ArmTrainingConfig(algorithm="rsl_rl_ppo"),
        device="cpu",
    )
    # 1001 // 24 == 41 (24 * 41 = 984; remainder 17)
    assert agent._iterations_for(1001) == 41
    # Edge case: total < num_steps_per_env -> 0 iterations (caller's responsibility)
    assert agent._iterations_for(10) == 0
    # Exact multiple
    assert agent._iterations_for(240) == 10


def test_rsl_rl_ppo_agent_device_sourced_from_config() -> None:
    """Constructing without an explicit device kwarg uses ppo_cfg.device.

    Regression: F1 tech-debt cleanup moved the hardcoded "cuda:0"
    default into RslRlPpoConfig.device so it is overridable via YAML
    overlay without touching the agent constructor.
    """
    from armdroid.config.schema.training import ArmTrainingConfig, RslRlPpoConfig
    from armdroid.control.rsl_rl_agent import RslRlPpoAgent

    agent = RslRlPpoAgent(
        ppo_cfg=RslRlPpoConfig(device="cpu"),
        training_cfg=ArmTrainingConfig(algorithm="rsl_rl_ppo"),
    )
    assert agent._device == "cpu"

    # Explicit kwarg overrides the config default.
    agent_override = RslRlPpoAgent(
        ppo_cfg=RslRlPpoConfig(device="cuda:1"),
        training_cfg=ArmTrainingConfig(algorithm="rsl_rl_ppo"),
        device="cpu",
    )
    assert agent_override._device == "cpu"
