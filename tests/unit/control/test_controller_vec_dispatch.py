"""ArmController dispatch between single-env build() and vec build_vec() (F1).

Covers the vec-path branches in ``ArmController.build_for_env`` and
``ArmController.train_policy`` plus the runtime ``TypeError`` guard
against unsupported env shapes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from armdroid.control.controller import ArmController
from armdroid.domain.protocols import (
    ArmEnvironmentProtocol,
    VecArmEnvironmentProtocol,
)


def _vec_env(num_envs: int = 4) -> Any:
    env = MagicMock(spec=VecArmEnvironmentProtocol)
    env.num_envs = num_envs
    env.as_runner_env.return_value = MagicMock(num_envs=num_envs)
    return env


def _single_env() -> Any:
    return MagicMock(spec=ArmEnvironmentProtocol)


def _vec_agent() -> Any:
    """Agent satisfying both ArmRLAgentProtocol and VecArmRLAgentProtocol."""
    agent = MagicMock()
    agent.is_built = False
    agent.is_trained = False
    return agent


def _single_only_agent() -> Any:
    """Agent that does NOT implement build_vec (single-env only)."""
    agent = MagicMock(spec=["build", "train", "predict", "save", "load",
                            "is_built", "is_trained"])
    agent.is_built = False
    agent.is_trained = False
    return agent


def test_build_for_env_dispatches_to_build_vec_when_vec_env() -> None:
    """Vec env + agent with build_vec -> build_vec called, not build."""
    agent = _vec_agent()
    controller = ArmController(agent=agent, primitives=MagicMock())
    env = _vec_env(num_envs=4)

    controller.build_for_env(env)

    agent.build_vec.assert_called_once_with(env)
    agent.build.assert_not_called()


def test_build_for_env_dispatches_to_build_when_single_env() -> None:
    """Single env + any agent -> build() called, not build_vec()."""
    agent = _vec_agent()
    controller = ArmController(agent=agent, primitives=MagicMock())
    env = _single_env()

    controller.build_for_env(env)

    agent.build.assert_called_once_with(env)
    agent.build_vec.assert_not_called()


def test_build_for_env_rejects_vec_env_when_agent_lacks_build_vec() -> None:
    """Vec env paired with a single-only agent must raise ValueError.

    The runtime hardening guard refuses to silently degrade to
    ``agent.build()`` because the vec env's torch-tensor returns are
    incompatible with the single-env protocol's numpy contract.
    Callers must either configure a vec-capable agent (rsl_rl_ppo) or
    set ``num_envs == 1`` to use the single-env path.
    (Updated per Gemini Code Assist review: isinstance-based check.)
    """
    agent = _single_only_agent()
    controller = ArmController(agent=agent, primitives=MagicMock())
    vec_env = _vec_env(num_envs=2)

    with pytest.raises(
        ValueError,
        match=r"does not implement VecArmRLAgentProtocol",
    ):
        controller.build_for_env(vec_env)
    agent.build.assert_not_called()


def test_build_for_env_passes_non_vec_env_to_single_env_build() -> None:
    """An env that's not a vec env falls through to ``agent.build(env)``.

    The controller trusts duck-typing on the single-env path: any
    object that isn't isinstance(env, VecArmEnvironmentProtocol) is
    handed to ``agent.build(env)`` and the agent surfaces any structural
    mismatch from there. This mirrors the pre-F1 contract.
    """
    agent = _vec_agent()
    controller = ArmController(agent=agent, primitives=MagicMock())

    class _PlainEnv:
        """Plain object - not a Mock spec, not a protocol member."""

    plain_env = _PlainEnv()
    controller.build_for_env(plain_env)

    agent.build.assert_called_once_with(plain_env)
    agent.build_vec.assert_not_called()


def test_build_for_env_idempotent_when_already_built() -> None:
    """build_for_env is a no-op if agent.is_built is already True."""
    agent = _vec_agent()
    agent.is_built = True
    controller = ArmController(agent=agent, primitives=MagicMock())

    controller.build_for_env(_vec_env(num_envs=4))

    agent.build.assert_not_called()
    agent.build_vec.assert_not_called()


def test_train_policy_uses_train_vec_after_vec_build(
    tmp_path: Path,
) -> None:
    """After build_vec, train_policy must call train_vec, not train."""
    agent = _vec_agent()
    agent.save.return_value = tmp_path / "weights.pt"
    controller = ArmController(agent=agent, primitives=MagicMock())
    controller.build_for_env(_vec_env(num_envs=4))

    # After build_vec, is_built must report True for train_policy to proceed.
    agent.is_built = True

    controller.train_policy(total_timesteps=10_000)

    agent.train_vec.assert_called_once_with(10_000)
    agent.train.assert_not_called()


def test_train_policy_uses_train_after_single_env_build(
    tmp_path: Path,
) -> None:
    """After single-env build, train_policy must call train(), not train_vec()."""
    agent = _vec_agent()
    agent.save.return_value = tmp_path / "weights.pt"
    controller = ArmController(agent=agent, primitives=MagicMock())
    controller.build_for_env(_single_env())

    agent.is_built = True

    controller.train_policy(total_timesteps=10_000)

    agent.train.assert_called_once_with(10_000)
    agent.train_vec.assert_not_called()
