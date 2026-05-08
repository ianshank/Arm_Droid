"""Tests for armdroid configuration models."""

from __future__ import annotations

import pytest

from armdroid.config.schema import (
    ArmConfig,
    ArmCurriculumConfig,
    ArmPerceptionConfig,
    ArmPlanningConfig,
    ArmSettings,
    ArmSimConfig,
    ArmTaskConfig,
    ArmTrainingConfig,
    LLMReplannerConfig,
)


class TestArmConfig:
    """Test ArmConfig Pydantic model."""

    def test_defaults(self) -> None:
        cfg = ArmConfig()
        assert cfg.dof == 6
        assert cfg.gripper_type == "parallel"
        assert len(cfg.home_position) == 6

    def test_custom_dof(self) -> None:
        cfg = ArmConfig(dof=4, home_position=[0.0, 0.0, 0.0, 0.0])
        assert cfg.dof == 4
        assert len(cfg.home_position) == 4

    def test_home_dof_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="home_position length"):
            ArmConfig(dof=6, home_position=[0.0, 0.0, 0.0])


class TestArmSimConfig:
    """Test ArmSimConfig Pydantic model."""

    def test_defaults(self) -> None:
        cfg = ArmSimConfig()
        assert cfg.timestep_s == 0.002
        assert cfg.domain_randomization is True
        assert cfg.mass_range_pct == 20.0

    def test_custom_randomization(self) -> None:
        cfg = ArmSimConfig(mass_range_pct=50.0, friction_range=0.5)
        assert cfg.mass_range_pct == 50.0
        assert cfg.friction_range == 0.5


class TestArmPerceptionConfig:
    """Test ArmPerceptionConfig Pydantic model."""

    def test_defaults(self) -> None:
        cfg = ArmPerceptionConfig()
        assert cfg.depth_camera_type == "realsense_d435i"
        assert cfg.pose_estimator == "pnp"
        assert cfg.yolo_confidence_threshold == 0.5

    def test_yolo_backend_locked_to_ultralytics(self) -> None:
        """The Hailo and auto backends were dropped — only ultralytics is valid."""
        cfg = ArmPerceptionConfig()
        assert cfg.yolo_backend == "ultralytics"

        with pytest.raises(ValueError, match="yolo_backend"):
            ArmPerceptionConfig(yolo_backend="hailo")  # type: ignore[arg-type]

    def test_mock_camera(self) -> None:
        cfg = ArmPerceptionConfig(depth_camera_type="mock")
        assert cfg.depth_camera_type == "mock"


class TestArmPlanningConfig:
    """Test ArmPlanningConfig Pydantic model."""

    def test_defaults(self) -> None:
        cfg = ArmPlanningConfig()
        assert cfg.planner_backend == "pyperplan"
        assert cfg.llm_replanner_enabled is False
        assert cfg.max_replan_attempts == 3
        assert cfg.llm_replanner is None

    def test_llm_replanner_attached(self) -> None:
        replanner = LLMReplannerConfig(enabled=True, backend="anthropic")
        cfg = ArmPlanningConfig(llm_replanner_enabled=True, llm_replanner=replanner)
        assert cfg.llm_replanner is not None
        assert cfg.llm_replanner.backend == "anthropic"


class TestArmTrainingConfig:
    """Test ArmTrainingConfig Pydantic model."""

    def test_defaults(self) -> None:
        cfg = ArmTrainingConfig()
        assert cfg.algorithm == "sac_her"
        assert cfg.learning_rate == 3e-4
        assert cfg.buffer_size == 1_000_000
        assert cfg.her_goal_selection == "future"
        assert cfg.reward_grasp == 0.1
        assert cfg.penalty_collision == -0.5
        assert cfg.seed == 42

    def test_custom_algorithm(self) -> None:
        cfg = ArmTrainingConfig(algorithm="ppo")
        assert cfg.algorithm == "ppo"


class TestArmCurriculumConfig:
    """Test ArmCurriculumConfig Pydantic model."""

    def test_defaults(self) -> None:
        cfg = ArmCurriculumConfig()
        assert cfg.enabled is True
        assert cfg.stages == [1, 2, 3, 5]
        assert cfg.promotion_threshold == 0.8
        assert cfg.warm_start is True


class TestArmTaskConfig:
    """Test ArmTaskConfig Pydantic model."""

    def test_defaults(self) -> None:
        cfg = ArmTaskConfig()
        assert cfg.task_type == "tower_of_hanoi"
        assert cfg.num_disks == 3
        assert cfg.num_pegs == 3
        assert len(cfg.peg_positions) == 3

    def test_peg_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="peg_positions length"):
            ArmTaskConfig(num_pegs=4, peg_positions=[[0, 0, 0], [1, 0, 0]])

    def test_basket_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="basket_positions length"):
            ArmTaskConfig(num_baskets=2, basket_positions=[[0, 0, 0]])

    def test_laundry_task(self) -> None:
        cfg = ArmTaskConfig(task_type="laundry_sorting")
        assert cfg.task_type == "laundry_sorting"


class TestArmSettings:
    """Test the root ArmSettings model."""

    def test_defaults_construct_complete_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ArmSettings() with no args produces a fully usable config.

        The autouse conftest sets ARMDROID_MOCK_HARDWARE=true; explicitly
        delete it here so the default-False path is exercised.
        """
        monkeypatch.delenv("ARMDROID_MOCK_HARDWARE", raising=False)
        cfg = ArmSettings()
        assert cfg.arm.dof == 6
        assert cfg.arm_perception.yolo_backend == "ultralytics"
        assert cfg.arm_planning.planner_backend == "pyperplan"
        assert cfg.arm_training.algorithm == "sac_her"
        assert cfg.arm_curriculum.enabled is True
        assert cfg.arm_task.task_type == "tower_of_hanoi"
        assert cfg.logging.level == "INFO"
        assert cfg.mock_hardware is False

    def test_mock_hardware_from_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ARMDROID_MOCK_HARDWARE env var sets mock_hardware=True."""
        monkeypatch.setenv("ARMDROID_MOCK_HARDWARE", "true")
        cfg = ArmSettings()
        assert cfg.mock_hardware is True

    def test_nested_env_var_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ARMDROID_ARM__DOF=7 overrides the nested arm.dof field."""
        monkeypatch.setenv("ARMDROID_ARM__DOF", "7")
        # ArmConfig has a validator requiring home_position length to match dof,
        # so we also override the home_position via env var to keep validation happy.
        monkeypatch.setenv(
            "ARMDROID_ARM__HOME_POSITION",
            "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
        )
        cfg = ArmSettings()
        assert cfg.arm.dof == 7
        assert len(cfg.arm.home_position) == 7


class TestLoadSettingsConvenienceFunction:
    """The module-level load_settings() delegates to the generic loader."""

    def test_load_settings_returns_arm_settings(self) -> None:
        from armdroid.config.schema import load_settings

        cfg = load_settings()
        assert isinstance(cfg, ArmSettings)
