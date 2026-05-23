"""Regression: new sub-configs load cleanly from YAML + tolerate unknowns.

Closes peer-review H4: the v1 plan widened ``ArmSettings`` with three
new sub-configs (``arm_vla``, ``arm_safety``, ``arm_interaction``) but
did not pin the YAML loader behaviour. This test:

* asserts ``config/default.yaml`` still loads.
* asserts ``config/examples/gemini.example.yaml`` materialises every
  new sub-config with the expected typed values.
* asserts unknown YAML keys are tolerated (``pydantic-settings`` v2
  default ``extra="ignore"``) so future config overlays do not need a
  coordinated schema update.
"""

from __future__ import annotations

from pathlib import Path

from armdroid.config.schema import ArmSettings, load_settings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_GEMINI = _REPO_ROOT / "config" / "examples" / "gemini.example.yaml"


def test_default_yaml_still_loads() -> None:
    """Loading without any overlay must still produce a complete settings tree."""
    settings = load_settings(config_dir=_REPO_ROOT / "config")
    assert isinstance(settings, ArmSettings)
    # New sub-configs are present via default_factory even without overlay
    assert settings.arm_vla.enabled is False
    assert settings.arm_safety.asimov_rule_set == "off"
    assert settings.arm_interaction.enabled is False


def test_gemini_example_overlay_materialises_new_sections() -> None:
    settings = load_settings(_EXAMPLE_GEMINI, config_dir=_REPO_ROOT / "config")

    # Perception widening
    assert settings.arm_perception.detector_kind == "yolo"
    assert settings.arm_perception.gemini_model == "gemini-robotics-er-1.6"
    assert settings.arm_perception.frame_buffer_seconds == 30.0

    # LLM replanner widening
    assert settings.arm_planning.llm_replanner is not None
    assert settings.arm_planning.llm_replanner.backend == "gemini"
    assert settings.arm_planning.llm_replanner.safety_tier == "standard"

    # New sub-configs
    assert settings.arm_vla.enabled is False
    assert settings.arm_vla.action_clip_rad == 0.25
    assert settings.arm_safety.asimov_rule_set == "off"
    assert settings.arm_safety.max_tcp_speed_mps == 0.5
    assert settings.arm_interaction.enabled is False
    assert settings.arm_interaction.audio_sample_rate_hz == 16000


def test_unknown_keys_in_yaml_are_ignored(tmp_path: Path) -> None:
    """pydantic-settings v2 ``extra="ignore"`` default: extra keys silently drop."""
    overlay = tmp_path / "extras.yaml"
    overlay.write_text("arm_vla:\n  enabled: true\n  some_future_field_not_in_schema: 42\n")
    settings = load_settings(overlay, config_dir=_REPO_ROOT / "config")
    assert settings.arm_vla.enabled is True
