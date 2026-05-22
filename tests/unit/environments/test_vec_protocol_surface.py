"""Surface tests for VecArmEnvironmentProtocol and VecArmRLAgentProtocol (F1)."""

from __future__ import annotations

from unittest.mock import MagicMock

from armdroid.domain.protocols import (
    VecArmEnvironmentProtocol,
    VecArmRLAgentProtocol,
)


def test_vec_env_protocol_is_runtime_checkable() -> None:
    """The vec protocol must be ``@runtime_checkable`` so isinstance works."""
    fake = MagicMock(spec=VecArmEnvironmentProtocol)
    fake.num_envs = 4
    assert isinstance(fake, VecArmEnvironmentProtocol)


def test_vec_env_protocol_required_methods() -> None:
    """The vec protocol declares reset, step, close, as_runner_env, num_envs."""
    expected = {"reset", "step", "close", "as_runner_env", "num_envs"}
    assert expected.issubset(set(dir(VecArmEnvironmentProtocol)))


def test_vec_rl_agent_protocol_is_runtime_checkable() -> None:
    """``VecArmRLAgentProtocol`` must be ``@runtime_checkable``."""
    fake = MagicMock(spec=VecArmRLAgentProtocol)
    assert isinstance(fake, VecArmRLAgentProtocol)


def test_vec_rl_agent_protocol_required_methods() -> None:
    """``VecArmRLAgentProtocol`` declares build_vec, train_vec, predict, save, load."""
    expected = {"build_vec", "train_vec", "predict", "save", "load"}
    assert expected.issubset(set(dir(VecArmRLAgentProtocol)))


def test_vec_protocols_in_module_all() -> None:
    """Both protocols are exported via ``armdroid.domain.protocols.__all__``."""
    from armdroid.domain import protocols as _p

    assert "VecArmEnvironmentProtocol" in _p.__all__
    assert "VecArmRLAgentProtocol" in _p.__all__
