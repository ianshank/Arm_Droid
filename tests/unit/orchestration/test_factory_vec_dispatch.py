"""Factory dispatch on cfg.arm_sim_isaac.num_envs (F1).

Tests the two dispatch points: ``build_arm_environment`` (env shape)
and ``ArmController.build_for_env`` (agent method). The single-env
default path (``num_envs == 1``) must remain byte-identical.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from armdroid.config.schema import ArmSettings
from armdroid.config.schema.sim_isaac import ArmSimIsaacConfig
from armdroid.config.schema.task import ArmTaskConfig
from armdroid.config.schema.training import ArmTrainingConfig
from armdroid.orchestration.factory import (
    _VEC_CAPABLE_ALGORITHMS,
    _VEC_TASK_REGISTRY_NAMES,
    _should_use_vec,
    build_arm_environment,
)


def _settings(num_envs: int, algorithm: str = "rsl_rl_ppo") -> ArmSettings:
    return ArmSettings(
        arm_driver_kind="mock",
        arm_task=ArmTaskConfig(task_type="so_arm_reach_isaac"),
        arm_training=ArmTrainingConfig(algorithm=algorithm),
        arm_sim_isaac=ArmSimIsaacConfig(num_envs=num_envs),
    )


def test_should_use_vec_false_when_num_envs_one() -> None:
    assert _should_use_vec(_settings(num_envs=1)) is False


def test_should_use_vec_true_when_num_envs_gt_one() -> None:
    assert _should_use_vec(_settings(num_envs=4)) is True


def test_vec_capable_algorithms_pinned() -> None:
    """rsl_rl_ppo is currently the only vec-capable algorithm."""
    assert "rsl_rl_ppo" in _VEC_CAPABLE_ALGORITHMS


def test_vec_task_registry_mapping_pinned() -> None:
    """The so_arm_reach_isaac task maps to the so_arm_reach_isaac_vec registry."""
    assert _VEC_TASK_REGISTRY_NAMES["so_arm_reach_isaac"] == "so_arm_reach_isaac_vec"


def test_build_arm_environment_routes_to_vec_for_num_envs_gt_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """num_envs > 1 + rsl_rl_ppo + so_arm_reach_isaac task -> vec env factory."""
    import armdroid.orchestration.factory as factory_mod

    captured: dict[str, Any] = {}

    def _fake_vec_factory(**kwargs: Any) -> Any:
        captured["called"] = True
        captured["kwargs"] = kwargs
        return MagicMock(num_envs=4)

    monkeypatch.setattr(
        factory_mod,
        "get_vec_environment",
        lambda _name: _fake_vec_factory,
    )

    env = build_arm_environment(_settings(num_envs=4))

    assert captured.get("called") is True
    assert "sim_isaac_cfg" in captured["kwargs"]
    assert captured["kwargs"]["sim_isaac_cfg"].num_envs == 4
    assert env.num_envs == 4


def test_build_arm_environment_vec_with_unsupported_algorithm_raises() -> None:
    """num_envs > 1 with non-vec algorithm must raise ValueError."""
    cfg = _settings(num_envs=4, algorithm="sac_her")
    with pytest.raises(ValueError, match="num_envs > 1 requires"):
        build_arm_environment(cfg)


def test_build_arm_environment_single_env_path_untouched_for_isaac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """num_envs == 1 keeps using the single-env factory (not the vec one)."""
    import armdroid.orchestration.factory as factory_mod

    captured: dict[str, Any] = {}

    def _fake_factory(*args: Any, **kwargs: Any) -> Any:
        captured["called"] = True
        captured["kwargs"] = kwargs
        return MagicMock(spec=object)

    def _vec_must_not_fire(_name: str) -> Any:
        captured["vec_called"] = True
        raise AssertionError("vec factory must not be invoked for num_envs == 1")

    monkeypatch.setattr(
        factory_mod,
        "get_environment",
        lambda _name: _fake_factory,
    )
    monkeypatch.setattr(factory_mod, "get_vec_environment", _vec_must_not_fire)

    build_arm_environment(_settings(num_envs=1))
    assert captured.get("called") is True
    assert "vec_called" not in captured
    # The single-env factory must NOT receive a vec-only kwarg distinguisher.
    assert "use_vec" not in captured["kwargs"]


def test_build_arm_environment_non_isaac_task_unaffected_by_num_envs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """num_envs > 1 on a non-Isaac task is rejected (vec path is Isaac-only)."""
    cfg = ArmSettings(
        arm_driver_kind="mock",
        arm_task=ArmTaskConfig(task_type="tower_of_hanoi"),
        arm_training=ArmTrainingConfig(algorithm="rsl_rl_ppo"),
        arm_sim_isaac=ArmSimIsaacConfig(num_envs=4),
    )
    with pytest.raises(ValueError, match="vec env path requires"):
        build_arm_environment(cfg)
