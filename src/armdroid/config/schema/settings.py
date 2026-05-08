"""Root settings model and YAML+env loader."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from armdroid.config.logging import LoggingConfig
from armdroid.config.schema.arm import ArmConfig
from armdroid.config.schema.perception import ArmPerceptionConfig
from armdroid.config.schema.planning import ArmPlanningConfig
from armdroid.config.schema.sim import ArmSimConfig
from armdroid.config.schema.task import ArmTaskConfig
from armdroid.config.schema.training import ArmCurriculumConfig, ArmTrainingConfig


class ArmSettings(BaseSettings):
    """Root settings for armdroid.

    All sub-configs default to constructed instances so ``ArmSettings()``
    produces a complete usable configuration without any YAML overlay.
    YAML files override individual fields; ``ARMDROID_*`` env vars
    override everything.
    """

    model_config = SettingsConfigDict(
        env_prefix="ARMDROID_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    arm: ArmConfig = Field(default_factory=ArmConfig)
    arm_sim: ArmSimConfig = Field(default_factory=ArmSimConfig)
    arm_perception: ArmPerceptionConfig = Field(default_factory=ArmPerceptionConfig)
    arm_planning: ArmPlanningConfig = Field(default_factory=ArmPlanningConfig)
    arm_training: ArmTrainingConfig = Field(default_factory=ArmTrainingConfig)
    arm_curriculum: ArmCurriculumConfig = Field(default_factory=ArmCurriculumConfig)
    arm_task: ArmTaskConfig = Field(default_factory=ArmTaskConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    mock_hardware: bool = Field(
        default=False,
        description="Use mock hardware drivers instead of real serial/USB devices.",
    )


def load_settings(*overlay_paths: Path, config_dir: Path | None = None) -> ArmSettings:
    """Convenience wrapper: load ArmSettings from YAML overlays + env vars.

    Args:
        overlay_paths: YAML files to merge on top of ``<config_dir>/default.yaml``.
        config_dir: Directory containing ``default.yaml``. Defaults to repo ``config/``.

    Returns:
        Fully resolved :class:`ArmSettings`.
    """
    from armdroid.config.loader import load_settings as _generic_load

    return _generic_load(
        *overlay_paths,
        settings_class=ArmSettings,
        config_dir=config_dir,
    )


__all__ = ["ArmSettings", "load_settings"]
