"""AppLauncher singleton coordination: vec env + driver must not double-boot Kit (F1)."""

from __future__ import annotations

from typing import Any

import pytest

from armdroid.config.schema.sim_isaac import ArmSimIsaacConfig
from armdroid.config.schema.task import ArmTaskConfig
from armdroid.config.schema.training import ArmTrainingConfig
from tests.helpers.fake_isaac_env import make_fake_isaac_env


@pytest.fixture
def fresh_app_state() -> Any:
    """Reset the singleton flag before and after each test."""
    from armdroid.hardware.isaac_sim import _app_state

    _app_state.reset_for_tests()
    yield _app_state
    _app_state.reset_for_tests()


def test_vec_env_marks_kit_launched_when_first(
    fresh_app_state: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Kit is not yet booted, the vec env marks it launched after build."""
    from armdroid.environments.isaac import reach_vec

    monkeypatch.setattr(
        reach_vec, "_build_isaac_env",
        lambda *_a, **_kw: make_fake_isaac_env(num_envs=4),
    )
    env = reach_vec.SoArmReachIsaacVecEnv(
        task_cfg=ArmTaskConfig(),
        training_cfg=ArmTrainingConfig(),
        sim_isaac_cfg=ArmSimIsaacConfig(num_envs=4),
    )
    env.reset(seed=0)
    assert fresh_app_state.is_app_launched() is True
    assert env._kit_booted_here is True


def test_vec_env_observes_already_launched_kit(
    fresh_app_state: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Kit was already booted (e.g. by the driver), vec env reuses it."""
    from armdroid.environments.isaac import reach_vec

    fresh_app_state.mark_launched()  # simulate driver-first ordering
    monkeypatch.setattr(
        reach_vec, "_build_isaac_env",
        lambda *_a, **_kw: make_fake_isaac_env(num_envs=4),
    )
    env = reach_vec.SoArmReachIsaacVecEnv(
        task_cfg=ArmTaskConfig(),
        training_cfg=ArmTrainingConfig(),
        sim_isaac_cfg=ArmSimIsaacConfig(num_envs=4),
    )
    env.reset(seed=0)

    assert fresh_app_state.is_app_launched() is True
    # Kit was already up; the vec env did not boot it here.
    assert env._kit_booted_here is False


def test_vec_env_kit_boot_span_emitted(
    fresh_app_state: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``SPAN_ENV_VEC_KIT_BOOT`` span fires around the lazy build."""
    from armdroid import telemetry as tel_mod
    from armdroid.environments.isaac import reach_vec
    from tests.helpers.recording_telemetry import RecordingTelemetry

    recorder = RecordingTelemetry()
    tel_mod.configure_telemetry(recorder)
    try:
        monkeypatch.setattr(
            reach_vec, "_build_isaac_env",
            lambda *_a, **_kw: make_fake_isaac_env(num_envs=2),
        )
        env = reach_vec.SoArmReachIsaacVecEnv(
            task_cfg=ArmTaskConfig(),
            training_cfg=ArmTrainingConfig(),
            sim_isaac_cfg=ArmSimIsaacConfig(num_envs=2),
        )
        env.reset(seed=0)
        # The reset() span contains the kit-boot span (lazy build inside _ensure_built).
        assert tel_mod.SPAN_ENV_VEC_KIT_BOOT in recorder.spans
    finally:
        tel_mod.configure_telemetry(None)
