"""Unit tests for ArmSimIsaacConfig + _default_sim_cfg().

Closes PR-B B.2. Asserts:
* default construction without args
* env-var override via ARMDROID_ARM_SIM_ISAAC__<FIELD>
* bounds violation rejection
* gripper convention defaults match armdroid (0=open, 1=closed)
* _default_sim_cfg() returns a fresh instance
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from armdroid.config.schema import ArmSettings, ArmSimIsaacConfig
from armdroid.config.schema.sim_isaac import _default_sim_cfg


class TestDefaultConstruction:
    def test_default_construction_succeeds(self) -> None:
        cfg = ArmSimIsaacConfig()
        assert isinstance(cfg, ArmSimIsaacConfig)

    def test_default_headless_is_true(self) -> None:
        assert ArmSimIsaacConfig().headless is True

    def test_default_num_envs_is_one(self) -> None:
        assert ArmSimIsaacConfig().num_envs == 1

    def test_default_physics_dt_is_5ms(self) -> None:
        cfg = ArmSimIsaacConfig()
        assert 0.0 < cfg.physics_dt_s <= 0.05
        assert cfg.physics_dt_s == 0.005

    def test_default_urdf_path_is_so101(self) -> None:
        cfg = ArmSimIsaacConfig()
        assert isinstance(cfg.urdf_fallback_path, Path)
        assert cfg.urdf_fallback_path.name == "so101_new_calib.urdf"

    def test_default_arm_joint_names_are_5(self) -> None:
        cfg = ArmSimIsaacConfig()
        assert len(cfg.arm_joint_names) == 5
        assert cfg.arm_joint_names == (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        )

    def test_default_gripper_joint_index_is_5(self) -> None:
        """Default gripper sits at the end of a 6-DoF (5 arm + 1 gripper) vector."""
        assert ArmSimIsaacConfig().gripper_joint_index == 5


class TestGripperConvention:
    """Closes R5 + R7 — armdroid normalised 0=open, 1=closed."""

    def test_gripper_normalised_open_is_zero(self) -> None:
        assert ArmSimIsaacConfig().gripper_normalised_open == 0.0

    def test_gripper_normalised_closed_is_one(self) -> None:
        assert ArmSimIsaacConfig().gripper_normalised_closed == 1.0

    def test_gripper_radians_open_positive(self) -> None:
        """URDF convention: positive radians = open jaw."""
        cfg = ArmSimIsaacConfig()
        assert cfg.gripper_joint_radians_open > cfg.gripper_joint_radians_closed


class TestBoundsValidation:
    def test_num_envs_negative_rejected(self) -> None:
        with pytest.raises(ValidationError, match="num_envs"):
            ArmSimIsaacConfig(num_envs=-1)

    def test_num_envs_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="num_envs"):
            ArmSimIsaacConfig(num_envs=0)

    def test_physics_dt_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="physics_dt_s"):
            ArmSimIsaacConfig(physics_dt_s=0.0)

    def test_decimation_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="decimation"):
            ArmSimIsaacConfig(decimation=0)

    def test_arm_effort_limit_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="arm_effort_limit_sim"):
            ArmSimIsaacConfig(arm_effort_limit_sim=0.0)

    def test_negative_stiffness_rejected(self) -> None:
        with pytest.raises(ValidationError, match="arm_stiffness_shoulder_pan"):
            ArmSimIsaacConfig(arm_stiffness_shoulder_pan=-1.0)

    def test_friction_range_inverted_rejected(self) -> None:
        with pytest.raises(ValidationError, match="friction_range_sim"):
            ArmSimIsaacConfig(friction_range_sim=(1.5, 0.5))

    def test_mass_scale_range_inverted_rejected(self) -> None:
        with pytest.raises(ValidationError, match="mass_scale_range"):
            ArmSimIsaacConfig(mass_scale_range=(1.2, 0.8))

    def test_valid_dr_ranges_accepted(self) -> None:
        cfg = ArmSimIsaacConfig(
            friction_range_sim=(0.5, 1.5),
            mass_scale_range=(0.8, 1.2),
        )
        assert cfg.friction_range_sim == (0.5, 1.5)
        assert cfg.mass_scale_range == (0.8, 1.2)

    def test_equal_range_endpoints_accepted(self) -> None:
        """Equal min/max (no randomization) should be valid."""
        cfg = ArmSimIsaacConfig(
            friction_range_sim=(1.0, 1.0),
            mass_scale_range=(1.0, 1.0),
        )
        assert cfg.friction_range_sim == (1.0, 1.0)


class TestEnvVarOverride:
    def test_env_var_overrides_num_envs_via_arm_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ARMDROID_ARM_SIM_ISAAC__NUM_ENVS=4 propagates through ArmSettings."""
        monkeypatch.setenv("ARMDROID_ARM_SIM_ISAAC__NUM_ENVS", "4")
        cfg = ArmSettings()
        assert cfg.arm_sim_isaac.num_envs == 4

    def test_env_var_overrides_headless(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ARMDROID_ARM_SIM_ISAAC__HEADLESS", "false")
        cfg = ArmSettings()
        assert cfg.arm_sim_isaac.headless is False


class TestArmSettingsIntegration:
    def test_arm_sim_isaac_present_on_root_settings(self) -> None:
        """ArmSettings.arm_sim_isaac is constructed by default."""
        settings = ArmSettings()
        assert isinstance(settings.arm_sim_isaac, ArmSimIsaacConfig)

    def test_arm_sim_isaac_round_trips_through_yaml_overlay(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """YAML overlay setting arm_sim_isaac.num_envs=8 reaches the construct."""
        from armdroid.config.schema import load_settings

        overlay = tmp_path / "overlay.yaml"
        overlay.write_text("arm_sim_isaac:\n  num_envs: 8\n  headless: false\n")
        # config_dir must contain default.yaml; reuse the repo's config/.
        repo_config = Path(__file__).resolve().parents[2] / "config"
        cfg = load_settings(overlay, config_dir=repo_config)
        assert cfg.arm_sim_isaac.num_envs == 8
        assert cfg.arm_sim_isaac.headless is False


class TestDefaultSimCfgFactory:
    def test_returns_fresh_instance(self) -> None:
        a = _default_sim_cfg()
        b = _default_sim_cfg()
        assert a is not b
        assert isinstance(a, ArmSimIsaacConfig)

    def test_returns_default_field_values(self) -> None:
        cfg = _default_sim_cfg()
        assert cfg.num_envs == 1
        assert cfg.headless is True
        assert cfg.gripper_joint_index == 5
