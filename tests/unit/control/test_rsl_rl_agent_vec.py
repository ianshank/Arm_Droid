"""Vec methods on RslRlPpoAgent: build_vec, train_vec (F1)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from armdroid import telemetry as tel_mod
from armdroid.config.schema.training import ArmTrainingConfig, RslRlPpoConfig
from armdroid.control.rsl_rl_agent import RslRlPpoAgent
from armdroid.domain.protocols import VecArmEnvironmentProtocol
from tests.helpers.recording_telemetry import RecordingTelemetry


@pytest.fixture
def recording_tel() -> Any:
    rec = RecordingTelemetry()
    tel_mod.configure_telemetry(rec)
    yield rec
    tel_mod.configure_telemetry(None)


def _vec_env(num_envs: int = 4) -> Any:
    env = MagicMock(spec=VecArmEnvironmentProtocol)
    env.num_envs = num_envs
    env.as_runner_env.return_value = MagicMock(num_envs=num_envs)
    return env


def test_build_vec_uses_as_runner_env(
    monkeypatch: pytest.MonkeyPatch, recording_tel: RecordingTelemetry,
) -> None:
    """``build_vec`` must call ``env.as_runner_env`` and instantiate the runner."""
    fake_runner = MagicMock()
    runner_class = MagicMock(return_value=fake_runner)
    monkeypatch.setattr(
        "armdroid.control.rsl_rl_agent._import_on_policy_runner",
        lambda: runner_class,
    )
    monkeypatch.setattr(
        RslRlPpoAgent,
        "_build_runner_cfg",
        lambda self: MagicMock(),
    )
    agent = RslRlPpoAgent(
        ppo_cfg=RslRlPpoConfig(),
        training_cfg=ArmTrainingConfig(algorithm="rsl_rl_ppo"),
        device="cpu",
    )
    env = _vec_env(num_envs=4)
    agent.build_vec(env)

    env.as_runner_env.assert_called_once()
    runner_class.assert_called_once()
    assert agent.is_built  # property reads ``self._runner is not None``
    assert tel_mod.SPAN_AGENT_BUILD_VEC in recording_tel.spans


def test_train_vec_calls_runner_learn(
    monkeypatch: pytest.MonkeyPatch, recording_tel: RecordingTelemetry,
) -> None:
    """``train_vec`` delegates to ``runner.learn`` and flips ``is_trained``."""
    fake_runner = MagicMock()
    runner_class = MagicMock(return_value=fake_runner)
    monkeypatch.setattr(
        "armdroid.control.rsl_rl_agent._import_on_policy_runner",
        lambda: runner_class,
    )
    monkeypatch.setattr(
        RslRlPpoAgent,
        "_build_runner_cfg",
        lambda self: MagicMock(),
    )
    agent = RslRlPpoAgent(
        ppo_cfg=RslRlPpoConfig(),
        training_cfg=ArmTrainingConfig(algorithm="rsl_rl_ppo"),
        device="cpu",
    )
    agent.build_vec(_vec_env(num_envs=4))
    agent.train_vec(total_timesteps=10_000)

    fake_runner.learn.assert_called_once()
    assert agent.is_trained
    assert tel_mod.SPAN_AGENT_TRAIN_VEC in recording_tel.spans


def test_train_vec_without_build_raises() -> None:
    """Calling ``train_vec`` before ``build_vec`` must raise RuntimeError."""
    agent = RslRlPpoAgent(
        ppo_cfg=RslRlPpoConfig(),
        training_cfg=ArmTrainingConfig(algorithm="rsl_rl_ppo"),
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="not built"):
        agent.train_vec()


def test_iterations_for_helper_extracted() -> None:
    """``_iterations_for`` must be a public-on-instance helper.

    Pure extraction of the inline computation previously in ``train()``;
    used by both ``train()`` and ``train_vec()``.
    """
    agent = RslRlPpoAgent(
        ppo_cfg=RslRlPpoConfig(),
        training_cfg=ArmTrainingConfig(algorithm="rsl_rl_ppo"),
        device="cpu",
    )
    # None -> ppo_cfg.num_iterations default
    assert agent._iterations_for(None) == agent._ppo_cfg.num_iterations
    # Explicit total_timesteps -> floor division by num_steps_per_env
    explicit = 10_000
    assert agent._iterations_for(explicit) == int(
        explicit // agent._ppo_cfg.num_steps_per_env,
    )
