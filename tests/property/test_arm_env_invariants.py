"""Property tests for ArmEnvironmentProtocol invariants.

Closes G2 (no env-protocol property tests despite a documented protocol).

Generates random action vectors via Hypothesis and verifies — for every
registered environment — that the returns of ``reset()`` and ``step()``
match the protocol's documented shape:

* ``reset()`` returns ``(dict, dict)`` with the expected goal-conditioned
  observation keys.
* ``step()`` returns the documented 5-tuple
  ``(dict, float, bool, bool, dict)`` — reward is a Python ``float``,
  not numpy/tensor, and ``terminated``/``truncated`` are Python ``bool``.

Verified by peer review (S6): both ``RewardShaper.compute()`` and the
two built-in ``_check_goal()`` impls already return the right Python
types, so this test should pass on the first GREEN. Type-coercion is
NOT expected to be needed; failures here indicate a regression.

PR-B extends ``_BUILTIN_ENVS`` to include ``"so_arm_reach_isaac"`` —
the Isaac branch is gated on ``pytest.importorskip("isaaclab")`` so
this file remains skip-clean without the optional extra installed.
"""

from __future__ import annotations

# Built-in envs that PR-A can safely exercise without optional extras.
# PR-B (this commit) extends with "so_arm_reach_isaac" gated on
# importlib.util.find_spec("isaaclab") — peer-review S-5: find_spec
# checks installation without import side-effects (raw try/except
# import would trigger AppLauncher singleton in some Isaac Lab
# configs).
import importlib.util as _importlib_util

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from armdroid.config.schema import ArmTaskConfig, ArmTrainingConfig
from armdroid.domain.protocols import ArmEnvironmentProtocol
from armdroid.environments.registry import get_environment

_BUILTIN_ENVS = ("tower_of_hanoi", "laundry_sorting")
_HAS_ISAAC = _importlib_util.find_spec("isaaclab") is not None
if _HAS_ISAAC:
    _BUILTIN_ENVS = (*_BUILTIN_ENVS, "so_arm_reach_isaac")


def _env_for(name: str, dof: int = 6) -> ArmEnvironmentProtocol:
    """Construct a registered env, overriding ``task_type`` to match name."""
    task_cfg = ArmTaskConfig(task_type=name)  # type: ignore[arg-type]
    env = get_environment(name)(task_cfg, ArmTrainingConfig(), dof=dof)
    # Belt-and-braces: every registered env must satisfy the protocol.
    assert isinstance(env, ArmEnvironmentProtocol)
    return env


@pytest.mark.parametrize("name", _BUILTIN_ENVS)
class TestEnvProtocolInvariants:
    def test_reset_returns_tuple_of_two_dicts(self, name: str) -> None:
        env = _env_for(name)
        obs, info = env.reset(seed=0)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        # Goal-conditioned envs must expose these canonical keys.
        for key in ("observation", "achieved_goal", "desired_goal"):
            assert key in obs, f"missing observation key: {key}"
            assert isinstance(obs[key], np.ndarray), f"{key} should be ndarray"
            assert obs[key].dtype == np.float64, f"{key} should be float64"

    def test_reset_is_deterministic_under_seed(self, name: str) -> None:
        env_a = _env_for(name)
        env_b = _env_for(name)
        obs_a, _ = env_a.reset(seed=42)
        obs_b, _ = env_b.reset(seed=42)
        for key in obs_a:
            np.testing.assert_array_equal(obs_a[key], obs_b[key])

    @given(
        action=st.lists(
            st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=6,
            max_size=6,
        )
    )
    @settings(
        max_examples=25,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_step_returns_documented_5_tuple(
        self,
        name: str,
        action: list[float],
    ) -> None:
        env = _env_for(name)
        env.reset(seed=0)
        result = env.step(np.asarray(action, dtype=np.float64))
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        # Documented shape per ArmEnvironmentProtocol.step:
        assert isinstance(obs, dict)
        assert isinstance(reward, float), (
            f"reward should be Python float, got {type(reward).__name__}"
        )
        assert isinstance(terminated, bool), (
            f"terminated should be Python bool, got {type(terminated).__name__}"
        )
        assert isinstance(truncated, bool), (
            f"truncated should be Python bool, got {type(truncated).__name__}"
        )
        assert isinstance(info, dict)

    def test_step_count_monotonic_until_reset(self, name: str) -> None:
        env = _env_for(name)
        env.reset(seed=0)
        zero_action = np.zeros(6, dtype=np.float64)
        # Use the env's _step_count private attribute as the most direct
        # observable: protocols don't surface it but every env tracks it.
        last = getattr(env, "_step_count", 0)
        for _ in range(5):
            env.step(zero_action)
            current = getattr(env, "_step_count", 0)
            assert current > last
            last = current
        env.reset(seed=0)
        assert getattr(env, "_step_count", 0) == 0
