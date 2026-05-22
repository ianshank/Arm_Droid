"""Reusable fake ``ManagerBasedRLEnv`` for vec-env tests (F1).

Promoted from inline ``_fake_isaac_env`` factories duplicated across
``test_reach_vec_env.py``, ``test_app_launcher_coord.py``,
``test_factory_vec_dispatch.py``, and ``test_vec_env_invariants.py``.

The shape parameters (``obs_dim``, ``action_dim``) default to the
SO-ARM 6-DoF layout but are explicit so future tests with different
arm geometries can opt in.

Usage::

    from tests.helpers.fake_isaac_env import make_fake_isaac_env

    fake = make_fake_isaac_env(num_envs=4)
    monkeypatch.setattr(reach_vec, "_build_isaac_env", lambda *a, **k: fake)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import torch


def make_fake_isaac_env(
    *,
    num_envs: int,
    obs_dim: int = 6,
    obs_key: str = "observation",
    dtype: torch.dtype = torch.float32,
) -> Any:
    """Return a ``MagicMock`` shaped like an isaaclab ``ManagerBasedRLEnv``.

    The mock's ``reset()`` returns ``({obs_key: zeros}, {})`` and ``step()``
    returns a 5-tuple of zero tensors plus an empty info dict, with every
    batched tensor having leading dim ``num_envs``.

    Args:
        num_envs: Required - number of parallel envs the fake reports.
        obs_dim: Observation feature dimension. Default 6 matches the
            SO-ARM 6-DoF joint state vector.
        obs_key: Single observation dict key. Default "observation"
            matches the goal-conditioned env convention.
        dtype: Torch dtype for observation tensors; reward/term/trunc
            use derived dtypes (``float32`` / ``bool``).

    Returns:
        A ``MagicMock`` exposing ``num_envs``, ``reset``, ``step``, and
        ``close`` - the minimal surface ``SoArmReachIsaacVecEnv`` and
        ``RslRlPpoAgent.build_vec`` consume.
    """
    env = MagicMock()
    env.num_envs = num_envs
    obs = {obs_key: torch.zeros(num_envs, obs_dim, dtype=dtype)}
    env.reset.return_value = (obs, {})
    env.step.return_value = (
        obs,
        torch.zeros(num_envs, dtype=torch.float32),
        torch.zeros(num_envs, dtype=torch.bool),
        torch.zeros(num_envs, dtype=torch.bool),
        {},
    )
    return env


__all__ = ["make_fake_isaac_env"]
