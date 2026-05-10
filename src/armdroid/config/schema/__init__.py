"""Root configuration schema for armdroid — single source of truth.

All values read from YAML config files or environment variables.
Nothing hardcoded elsewhere. New fields MUST have defaults (backwards
compatibility guarantee).

This package re-exports every public symbol previously defined in
``armdroid.config.schema`` (when it was a single module). Existing imports
of the form ``from armdroid.config.schema import X`` continue to work
unchanged.

Submodules:
    arm: Robot arm hardware (joints, servos, transport, firmware, ArmConfig)
    llm: LLM replanner sub-config
    sim: MuJoCo simulation
    perception: Perception stack
    planning: Symbolic planner
    training: RL training + curriculum
    task: Task-specific (Hanoi, laundry sorting)
    settings: Root :class:`ArmSettings` and :func:`load_settings`
"""

from __future__ import annotations

from armdroid.config.schema.arm import (
    ArmConfig,
    ArmFirmwareConfig,
    ArmServoConfig,
    ArmTransportConfig,
    BleTransportConfig,
    JointLimits,
    TcpTransportConfig,
    TransportAuthConfig,
)
from armdroid.config.schema.llm import LLMReplannerConfig
from armdroid.config.schema.perception import ArmPerceptionConfig
from armdroid.config.schema.planning import ArmPlanningConfig
from armdroid.config.schema.settings import ArmSettings, load_settings
from armdroid.config.schema.sim import ArmSimConfig
from armdroid.config.schema.sim_isaac import ArmSimIsaacConfig
from armdroid.config.schema.task import ArmTaskConfig
from armdroid.config.schema.training import (
    ArmCurriculumConfig,
    ArmTrainingConfig,
    RslRlPpoConfig,
)

__all__ = [
    "ArmConfig",
    "ArmCurriculumConfig",
    "ArmFirmwareConfig",
    "ArmPerceptionConfig",
    "ArmPlanningConfig",
    "ArmServoConfig",
    "ArmSettings",
    "ArmSimConfig",
    "ArmSimIsaacConfig",
    "ArmTaskConfig",
    "ArmTrainingConfig",
    "ArmTransportConfig",
    "BleTransportConfig",
    "JointLimits",
    "LLMReplannerConfig",
    "RslRlPpoConfig",
    "TcpTransportConfig",
    "TransportAuthConfig",
    "load_settings",
]
