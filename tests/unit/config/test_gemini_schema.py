"""Unit tests for Phase A configuration widening.

Pins the new sub-configs (``ArmVlaConfig``, ``ArmSafetyConfig``,
``ArmInteractionConfig``), the widened ``LLMReplannerConfig.backend``
literal, and the new perception fields. Defaults must keep every legacy
path disabled so backwards-compat checkpoints in plan section 8 hold.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from armdroid.config.schema import (
    ArmInteractionConfig,
    ArmPerceptionConfig,
    ArmSafetyConfig,
    ArmSettings,
    ArmVlaConfig,
    LLMReplannerConfig,
)

# ---------------------------------------------------------------------------
# LLMReplannerConfig: gemini backend + new envelope fields
# ---------------------------------------------------------------------------


def test_llm_backend_literal_includes_gemini() -> None:
    cfg = LLMReplannerConfig(backend="gemini")
    assert cfg.backend == "gemini"


def test_llm_backend_literal_includes_openai_compat() -> None:
    cfg = LLMReplannerConfig(backend="openai_compat")
    assert cfg.backend == "openai_compat"


def test_llm_backend_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        LLMReplannerConfig(backend="grok")  # type: ignore[arg-type]


def test_llm_envelope_defaults() -> None:
    cfg = LLMReplannerConfig()
    assert cfg.api_endpoint == ""
    assert cfg.safety_tier == "standard"
    assert cfg.transport == "rest"


# ---------------------------------------------------------------------------
# ArmPerceptionConfig: new detector + buffer + Gemini fields
# ---------------------------------------------------------------------------


def test_perception_detector_kind_defaults_yolo() -> None:
    cfg = ArmPerceptionConfig()
    assert cfg.detector_kind == "yolo"


def test_perception_detector_kind_accepts_gemini() -> None:
    cfg = ArmPerceptionConfig(detector_kind="gemini_er")
    assert cfg.detector_kind == "gemini_er"


def test_perception_frame_buffer_bounds() -> None:
    with pytest.raises(ValidationError):
        ArmPerceptionConfig(frame_buffer_seconds=-1.0)
    with pytest.raises(ValidationError):
        ArmPerceptionConfig(frame_buffer_seconds=121.0)
    with pytest.raises(ValidationError):
        ArmPerceptionConfig(frame_buffer_fps=0)
    with pytest.raises(ValidationError):
        ArmPerceptionConfig(frame_buffer_fps=31.0)


def test_perception_open_vocab_max_queries_bounds() -> None:
    with pytest.raises(ValidationError):
        ArmPerceptionConfig(open_vocab_max_queries=0)
    with pytest.raises(ValidationError):
        ArmPerceptionConfig(open_vocab_max_queries=65)


def test_perception_pointcloud_bounds() -> None:
    with pytest.raises(ValidationError):
        ArmPerceptionConfig(pointcloud_subsample_stride=0)
    with pytest.raises(ValidationError):
        ArmPerceptionConfig(pointcloud_max_points=200000)


# ---------------------------------------------------------------------------
# ArmVlaConfig
# ---------------------------------------------------------------------------


def test_vla_defaults_disabled() -> None:
    cfg = ArmVlaConfig()
    assert cfg.enabled is False
    assert cfg.model == "gemini-robotics-er-1.6"
    assert cfg.api_key_env_var == "GEMINI_API_KEY"


def test_vla_action_clip_bounds() -> None:
    with pytest.raises(ValidationError):
        ArmVlaConfig(action_clip_rad=0.0)
    with pytest.raises(ValidationError):
        ArmVlaConfig(action_clip_rad=4.0)


def test_vla_frame_stack_bounds() -> None:
    with pytest.raises(ValidationError):
        ArmVlaConfig(frame_stack=0)


# ---------------------------------------------------------------------------
# ArmSafetyConfig
# ---------------------------------------------------------------------------


def test_safety_defaults_disabled() -> None:
    cfg = ArmSafetyConfig()
    assert cfg.asimov_rule_set == "off"
    assert cfg.neiss_corpus_path == Path("assets/safety/neiss_corpus.json")


def test_safety_velocity_bounds() -> None:
    with pytest.raises(ValidationError):
        ArmSafetyConfig(max_tcp_speed_mps=0.0)
    with pytest.raises(ValidationError):
        ArmSafetyConfig(max_joint_velocity_rad_s=11.0)


# ---------------------------------------------------------------------------
# ArmInteractionConfig
# ---------------------------------------------------------------------------


def test_interaction_defaults_disabled() -> None:
    cfg = ArmInteractionConfig()
    assert cfg.enabled is False
    assert cfg.backend == "null"


def test_interaction_audio_sample_rate_bounds() -> None:
    with pytest.raises(ValidationError):
        ArmInteractionConfig(audio_sample_rate_hz=0)
    with pytest.raises(ValidationError):
        ArmInteractionConfig(audio_sample_rate_hz=200000)


# ---------------------------------------------------------------------------
# ArmSettings: new sub-configs wired with default_factory
# ---------------------------------------------------------------------------


def test_arm_settings_default_factory_wires_new_subconfigs() -> None:
    settings = ArmSettings()
    assert isinstance(settings.arm_vla, ArmVlaConfig)
    assert isinstance(settings.arm_safety, ArmSafetyConfig)
    assert isinstance(settings.arm_interaction, ArmInteractionConfig)
    # All disabled by default to preserve byte-identical defaults
    assert settings.arm_vla.enabled is False
    assert settings.arm_safety.asimov_rule_set == "off"
    assert settings.arm_interaction.enabled is False


def test_arm_settings_legacy_kwargs_still_accepted() -> None:
    """Existing call sites (``mock_hardware=True``, ``arm_driver_kind="mock"``)
    must not break under the widened settings model.
    """
    settings = ArmSettings(arm_driver_kind="mock")
    assert settings.arm_driver_kind == "mock"
    settings_b = ArmSettings(mock_hardware=True, arm_driver_kind=None)
    assert settings_b.mock_hardware is True
