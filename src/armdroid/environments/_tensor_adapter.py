"""Generic torch ↔ numpy bridge for env wrappers (PR-B B.7).

Closes R3 from PR #8 review — Isaac Lab's ``ManagerBasedRLEnv`` returns
``torch.Tensor`` for reward / termination / observations, but
``ArmEnvironmentProtocol`` returns Python ``float`` / ``bool`` /
``NDArray[np.float64]``. ``_TensorAdapter`` is the *single* place that
does the conversion, with defensive ``.reshape(-1)[0]`` against the
``(num_envs,)`` vs ``(num_envs, 1)`` reward-shape ambiguity.

Module path is ``armdroid.environments._tensor_adapter`` (NOT
``armdroid.environments.isaac.common``) so its tests count toward the
85% coverage gate. Was moved out of the Isaac coverage-omit pattern
per peer-review C-4.

No isaaclab dep — torch is in armdroid's base ``[project.dependencies]``
so this module is always importable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray


class _TensorAdapter:
    """Bridge Isaac Lab ``ManagerBasedRLEnv`` tensor returns ↔ ArmEnvironmentProtocol numpy returns.

    For the protocol path, requires ``num_envs == 1`` so the wrapped
    env's vectorised returns can be unwrapped to scalar Python types.
    Vectorised training (``num_envs > 1``) bypasses this adapter and
    hits the raw ``ManagerBasedRLEnv`` directly via Isaac Lab's runner.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        dtype: type[np.floating[Any]] = np.float64,
    ) -> None:
        if num_envs > 1:
            msg = (
                f"Vectorised num_envs={num_envs} not supported by "
                "_TensorAdapter; use the raw ManagerBasedRLEnv via "
                "SoArmReachIsaacEnv._isaac_env for parallel rollout."
            )
            raise RuntimeError(msg)
        self._num_envs = num_envs
        self._dtype = dtype

    def step_to_protocol(
        self,
        out: tuple[Any, Any, Any, Any, Any],
    ) -> tuple[dict[str, NDArray[np.float64]], float, bool, bool, dict[str, Any]]:
        """Convert ManagerBasedRLEnv ``step`` 5-tuple to protocol shape.

        Defensive ``.reshape(-1)[0]`` handles both ``(num_envs,)`` and
        ``(num_envs, 1)`` reward shapes — Isaac Lab is documented as
        returning ``(num_envs,)`` but the ambiguity is real per
        upstream issue tracker.
        """
        obs, reward, terminated, truncated, info = out
        obs_np = self._to_numpy_dict(obs)

        reward_arr = np.asarray(self._to_numpy(reward))
        reward_f = float(reward_arr.reshape(-1)[0])
        if not math.isfinite(reward_f):
            msg = f"reward is non-finite: {reward_f}"
            raise ValueError(msg)

        term_arr = np.asarray(self._to_numpy(terminated))
        terminated_b = bool(term_arr.reshape(-1)[0])

        trunc_arr = np.asarray(self._to_numpy(truncated))
        truncated_b = bool(trunc_arr.reshape(-1)[0])

        return obs_np, reward_f, terminated_b, truncated_b, info

    def reset_to_protocol(
        self,
        out: tuple[Any, Any],
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
        """Convert ``ManagerBasedRLEnv.reset`` (obs_dict, info) to protocol shape."""
        obs, info = out
        return self._to_numpy_dict(obs), info

    def action_from_protocol(self, action: NDArray[np.float64]) -> Any:
        """Convert numpy ``(dof,)`` action to torch ``(num_envs, dof)`` tensor."""
        import torch

        return torch.tensor(action, dtype=torch.float32).reshape(self._num_envs, -1)

    def _to_numpy(self, t: Any) -> NDArray[np.float64]:
        """Convert any tensor / ndarray to dtype-cast numpy."""
        arr = t.cpu().numpy() if hasattr(t, "cpu") else t
        return np.asarray(arr, dtype=self._dtype)

    def _to_numpy_dict(self, d: Mapping[str, Any]) -> dict[str, NDArray[np.float64]]:
        """Convert a Mapping[str, tensor-or-ndarray] to dict[str, ndarray]."""
        return {k: self._to_numpy(v) for k, v in d.items()}


__all__ = ["_TensorAdapter"]
