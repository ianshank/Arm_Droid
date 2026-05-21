"""ArmController dispatch between single-env build() and vec build_vec() (F1).

Covers the vec-path branches in ``ArmController.build_for_env`` and
``ArmController.train_policy`` plus the runtime ``ValueError`` raised
when a vec env is paired with a non-vec agent.

Uses concrete test-double classes rather than ``MagicMock(spec=...)``
because Python 3.12's stricter ``runtime_checkable`` Protocol semantics
do not always recognise bare or spec'd MagicMocks as protocol members.
The doubles satisfy the protocols structurally and expose ``MagicMock``
sub-attributes for call-count / argument assertions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from armdroid.control.controller import ArmController
from armdroid.domain.protocols import (
    ArmEnvironmentProtocol,
    VecArmEnvironmentProtocol,
    VecArmRLAgentProtocol,
)


class _SingleEnvDouble:
    """Concrete double for :class:`ArmEnvironmentProtocol`."""

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        return {}, {}

    def step(
        self, action: NDArray[np.float64],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return {}, 0.0, False, False, {}

    def render(self) -> NDArray[np.uint8] | None:
        return None

    def close(self) -> None:
        return None


class _VecEnvDouble:
    """Concrete double for :class:`VecArmEnvironmentProtocol`."""

    def __init__(self, num_envs: int = 4) -> None:
        self._num_envs = num_envs
        self.as_runner_env_mock = MagicMock(return_value=MagicMock(num_envs=num_envs))

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        return {}, {}

    def step(
        self, action: Any,
    ) -> tuple[dict[str, Any], Any, Any, Any, dict[str, Any]]:
        return {}, None, None, None, {}

    def close(self) -> None:
        return None

    def as_runner_env(self) -> Any:
        return self.as_runner_env_mock()


class _SingleOnlyAgentDouble:
    """Single-env agent - satisfies ArmRLAgentProtocol only, NOT vec."""

    def __init__(self) -> None:
        self.build = MagicMock()
        self.train = MagicMock()
        self.predict = MagicMock(return_value=np.zeros(6, dtype=np.float64))
        self.save = MagicMock(return_value=Path("weights.pt"))
        self.load = MagicMock()
        self.is_built = False
        self.is_trained = False


class _VecCapableAgentDouble:
    """Agent satisfying BOTH ArmRLAgentProtocol and VecArmRLAgentProtocol."""

    def __init__(self) -> None:
        self.build = MagicMock()
        self.build_vec = MagicMock()
        self.train = MagicMock()
        self.train_vec = MagicMock()
        self.predict = MagicMock(return_value=np.zeros(6, dtype=np.float64))
        self.save = MagicMock(return_value=Path("weights.pt"))
        self.load = MagicMock()
        self.is_built = False
        self.is_trained = False


def _vec_env(num_envs: int = 4) -> _VecEnvDouble:
    return _VecEnvDouble(num_envs=num_envs)


def _single_env() -> _SingleEnvDouble:
    return _SingleEnvDouble()


def _vec_agent() -> _VecCapableAgentDouble:
    return _VecCapableAgentDouble()


def _single_only_agent() -> _SingleOnlyAgentDouble:
    return _SingleOnlyAgentDouble()


def test_test_doubles_satisfy_their_protocols() -> None:
    """Sanity check the test doubles match the runtime_checkable protocols.

    If a future protocol change adds a member, this test fails fast
    with a clear "doesn't satisfy" error instead of obscure dispatch
    failures in the other tests below.
    """
    assert isinstance(_single_env(), ArmEnvironmentProtocol)
    assert isinstance(_vec_env(), VecArmEnvironmentProtocol)
    assert isinstance(_vec_agent(), VecArmRLAgentProtocol)
    assert not isinstance(_single_only_agent(), VecArmRLAgentProtocol)


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
