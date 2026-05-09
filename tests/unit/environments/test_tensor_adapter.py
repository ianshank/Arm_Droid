"""Unit tests for _TensorAdapter (PR-B B.7).

Uses real torch (already in base deps per pyproject.toml) so the
production ``t.cpu().numpy()`` code path is exercised. Numpy stubs
would NOT have ``.cpu()`` and would silently miss the branch.

Closes R3 from PR #8 review (tensor → scalar protocol conversion).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from armdroid.environments._tensor_adapter import _TensorAdapter


class TestConstruction:
    def test_default_num_envs_one(self) -> None:
        adapter = _TensorAdapter(num_envs=1)
        assert adapter._num_envs == 1

    def test_num_envs_gt_one_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Vectorised num_envs=2"):
            _TensorAdapter(num_envs=2)

    def test_num_envs_huge_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Vectorised num_envs=1024"):
            _TensorAdapter(num_envs=1024)


class TestStepToProtocol:
    def test_unwraps_torch_tensors_to_python_scalars(self) -> None:
        adapter = _TensorAdapter(num_envs=1)
        obs = {
            "observation": torch.zeros((1, 7)),
            "achieved_goal": torch.zeros((1, 3)),
            "desired_goal": torch.zeros((1, 3)),
        }
        out = (
            obs,
            torch.tensor([0.5]),
            torch.tensor([False]),
            torch.tensor([False]),
            {"info": "x"},
        )
        obs_np, reward, term, trunc, info = adapter.step_to_protocol(out)
        assert isinstance(reward, float)
        assert reward == 0.5
        assert isinstance(term, bool)
        assert term is False
        assert isinstance(trunc, bool)
        assert trunc is False
        assert obs_np["observation"].dtype == np.float64
        assert obs_np["observation"].shape == (1, 7)
        assert info == {"info": "x"}

    def test_handles_2d_reward_shape(self) -> None:
        """Defensive .reshape(-1)[0] handles (num_envs=1, 1) reward shape."""
        adapter = _TensorAdapter(num_envs=1)
        # Use 0.5 (exact in float32) so torch.float32 -> float64 round-trip
        # is bit-exact. Other values would need pytest.approx because torch
        # tensors default to float32.
        out = (
            {},
            torch.tensor([[0.5]]),
            torch.tensor([False]),
            torch.tensor([False]),
            {},
        )
        _, reward, _, _, _ = adapter.step_to_protocol(out)
        assert reward == 0.5

    def test_raises_on_nan_reward(self) -> None:
        adapter = _TensorAdapter(num_envs=1)
        out = (
            {},
            torch.tensor([float("nan")]),
            torch.tensor([False]),
            torch.tensor([False]),
            {},
        )
        with pytest.raises(ValueError, match="non-finite"):
            adapter.step_to_protocol(out)

    def test_raises_on_inf_reward(self) -> None:
        adapter = _TensorAdapter(num_envs=1)
        out = (
            {},
            torch.tensor([float("inf")]),
            torch.tensor([False]),
            torch.tensor([False]),
            {},
        )
        with pytest.raises(ValueError, match="non-finite"):
            adapter.step_to_protocol(out)


class TestResetToProtocol:
    def test_unwraps_tuple(self) -> None:
        adapter = _TensorAdapter(num_envs=1)
        obs = {"observation": torch.zeros((1, 7))}
        info = {"task": "reach"}
        obs_np, info_out = adapter.reset_to_protocol((obs, info))
        assert isinstance(obs_np, dict)
        assert obs_np["observation"].dtype == np.float64
        assert obs_np["observation"].shape == (1, 7)
        assert info_out == info

    def test_preserves_obs_keys(self) -> None:
        adapter = _TensorAdapter(num_envs=1)
        obs = {
            "observation": torch.zeros((1, 7)),
            "achieved_goal": torch.zeros((1, 3)),
            "desired_goal": torch.ones((1, 3)),
        }
        obs_np, _ = adapter.reset_to_protocol((obs, {}))
        assert set(obs_np.keys()) == {"observation", "achieved_goal", "desired_goal"}


class TestActionFromProtocol:
    def test_returns_torch_tensor(self) -> None:
        adapter = _TensorAdapter(num_envs=1)
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
        out = adapter.action_from_protocol(action)
        assert hasattr(out, "cpu")  # is a tensor
        assert tuple(out.shape) == (1, 6)
        assert out.dtype == torch.float32

    def test_uses_num_envs_from_config(self) -> None:
        """num_envs comes from the adapter's config, not hardcoded."""
        adapter = _TensorAdapter(num_envs=1)
        action = np.zeros(6, dtype=np.float64)
        out = adapter.action_from_protocol(action)
        assert tuple(out.shape) == (1, 6)


class TestToNumpyHelpers:
    def test_to_numpy_passes_through_ndarray(self) -> None:
        """ndarray input (no .cpu()) is cast to dtype but not modified."""
        adapter = _TensorAdapter(num_envs=1)
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = adapter._to_numpy(arr)
        np.testing.assert_array_equal(out, arr)
        assert out.dtype == np.float64  # cast to float64

    def test_to_numpy_unwraps_torch_tensor(self) -> None:
        adapter = _TensorAdapter(num_envs=1)
        t = torch.tensor([1.0, 2.0, 3.0])
        out = adapter._to_numpy(t)
        np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))
        assert out.dtype == np.float64

    def test_to_numpy_dict_preserves_keys(self) -> None:
        adapter = _TensorAdapter(num_envs=1)
        d = {"a": torch.tensor([1.0]), "b": np.array([2.0])}
        out = adapter._to_numpy_dict(d)
        assert set(out.keys()) == {"a", "b"}
        assert out["a"].dtype == np.float64
        assert out["b"].dtype == np.float64
