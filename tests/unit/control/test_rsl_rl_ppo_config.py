"""Unit tests for RslRlPpoConfig (PR-B B.3)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from armdroid.config.schema import ArmSettings, ArmTrainingConfig, RslRlPpoConfig


class TestDefaultConstruction:
    def test_default_construction_succeeds(self) -> None:
        cfg = RslRlPpoConfig()
        assert isinstance(cfg, RslRlPpoConfig)

    def test_default_num_iterations_positive(self) -> None:
        assert RslRlPpoConfig().num_iterations == 1000

    def test_default_learning_rate_positive(self) -> None:
        cfg = RslRlPpoConfig()
        assert cfg.learning_rate == 1e-3
        assert cfg.learning_rate > 0.0

    def test_default_actor_critic_dims(self) -> None:
        cfg = RslRlPpoConfig()
        assert cfg.actor_hidden_dims == (64, 64)
        assert cfg.critic_hidden_dims == (64, 64)

    def test_default_activation(self) -> None:
        assert RslRlPpoConfig().activation == "elu"

    def test_default_schedule_is_adaptive(self) -> None:
        assert RslRlPpoConfig().schedule == "adaptive"

    def test_default_checkpoint_is_none(self) -> None:
        assert RslRlPpoConfig().checkpoint_path is None


class TestBoundsValidation:
    def test_num_steps_per_env_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="num_steps_per_env"):
            RslRlPpoConfig(num_steps_per_env=0)

    def test_num_iterations_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="num_iterations"):
            RslRlPpoConfig(num_iterations=0)

    def test_learning_rate_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="learning_rate"):
            RslRlPpoConfig(learning_rate=0.0)

    def test_gamma_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError, match="gamma"):
            RslRlPpoConfig(gamma=1.5)

    def test_gae_lambda_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="gae_lambda"):
            RslRlPpoConfig(gae_lambda=0.0)

    def test_clip_param_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="clip_param"):
            RslRlPpoConfig(clip_param=0.0)

    def test_negative_entropy_coef_rejected(self) -> None:
        with pytest.raises(ValidationError, match="entropy_coef"):
            RslRlPpoConfig(entropy_coef=-0.01)

    def test_invalid_activation_rejected(self) -> None:
        with pytest.raises(ValidationError, match="activation"):
            RslRlPpoConfig(activation="gelu")  # type: ignore[arg-type]


class TestArmTrainingAlgorithmExtension:
    """ArmTrainingConfig.algorithm Literal accepts 'rsl_rl_ppo' (PR-B)."""

    def test_rsl_rl_ppo_is_valid_algorithm(self) -> None:
        cfg = ArmTrainingConfig(algorithm="rsl_rl_ppo")
        assert cfg.algorithm == "rsl_rl_ppo"

    def test_existing_algorithms_still_valid(self) -> None:
        for algo in ("sac", "ppo", "sac_her"):
            cfg = ArmTrainingConfig(algorithm=algo)  # type: ignore[arg-type]
            assert cfg.algorithm == algo


class TestEnvVarOverride:
    def test_env_var_overrides_num_iterations(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ARMDROID_ARM_RSL_RL_PPO__NUM_ITERATIONS", "5000")
        cfg = ArmSettings()
        assert cfg.arm_rsl_rl_ppo.num_iterations == 5000

    def test_env_var_overrides_checkpoint_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        ckpt = tmp_path / "model.pt"
        monkeypatch.setenv("ARMDROID_ARM_RSL_RL_PPO__CHECKPOINT_PATH", str(ckpt))
        cfg = ArmSettings()
        assert cfg.arm_rsl_rl_ppo.checkpoint_path == ckpt


class TestArmSettingsIntegration:
    def test_arm_rsl_rl_ppo_present_on_root_settings(self) -> None:
        settings = ArmSettings()
        assert isinstance(settings.arm_rsl_rl_ppo, RslRlPpoConfig)
