"""Shared invariants for VecArmEnvironmentProtocol property tests (F1).

Lives alongside :mod:`tests.property.test_arm_env_invariants`. New
invariants that apply to both single-env and vec env should land in a
sibling helper module rather than being duplicated across test files.
"""

from __future__ import annotations

from typing import Any


def assert_reset_shape(
    obs: dict[str, Any], info: dict[str, Any], *, num_envs: int,
) -> None:
    """Assert the shape contract of a vec env ``reset()`` return.

    Args:
        obs: Observation dict (every value's leading dim must be ``num_envs``).
        info: Info dict (must be a dict; contents are env-specific).
        num_envs: Expected leading dim.
    """
    assert isinstance(obs, dict)
    assert isinstance(info, dict)
    for key, value in obs.items():
        assert value.shape[0] == num_envs, (
            f"obs[{key!r}] leading dim {value.shape[0]} != num_envs {num_envs}"
        )


def assert_step_shapes(
    obs: dict[str, Any],
    reward: Any,
    terminated: Any,
    truncated: Any,
    info: dict[str, Any],
    *,
    num_envs: int,
) -> None:
    """Assert the shape contract of a vec env ``step()`` 5-tuple."""
    assert isinstance(obs, dict)
    assert isinstance(info, dict)
    assert reward.shape == (num_envs,), (
        f"reward.shape {reward.shape} != ({num_envs},)"
    )
    assert terminated.shape == (num_envs,), (
        f"terminated.shape {terminated.shape} != ({num_envs},)"
    )
    assert truncated.shape == (num_envs,), (
        f"truncated.shape {truncated.shape} != ({num_envs},)"
    )
    for key, value in obs.items():
        assert value.shape[0] == num_envs, (
            f"obs[{key!r}] leading dim {value.shape[0]} != num_envs {num_envs}"
        )


__all__ = ["assert_reset_shape", "assert_step_shapes"]
