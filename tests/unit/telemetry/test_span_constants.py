"""Pin Phase A telemetry SPAN constant string values.

Closes peer-review H5: the v1 plan said "stable values" without
specifying them, making the regression brittle. This test asserts the
exact strings so any future drift is caught at CI time. The
``armdroid.*`` prefix is preserved for consistency with the
pre-existing ``SPAN_DRIVER_*`` / ``SPAN_ENV_*`` / ``SPAN_AGENT_*``
families.
"""

from __future__ import annotations

import armdroid.telemetry as tel

# (constant_name, expected_value)
_EXPECTED: tuple[tuple[str, str], ...] = (
    ("SPAN_PERCEPTION_DETECT", "armdroid.perception.detect"),
    ("SPAN_PERCEPTION_DETECT_TEXT", "armdroid.perception.detect_text"),
    ("SPAN_PERCEPTION_SCENE_REASON", "armdroid.perception.scene_reason"),
    (
        "SPAN_PERCEPTION_FRAME_BUFFER_APPEND",
        "armdroid.perception.frame_buffer_append",
    ),
    ("SPAN_LLM_REPLAN_REQUEST", "armdroid.llm_replan.request"),
    ("SPAN_LLM_REPLAN_PARSE", "armdroid.llm_replan.parse"),
    ("SPAN_LLM_REPLAN_FALLBACK", "armdroid.llm_replan.fallback"),
    ("SPAN_VLA_BUILD", "armdroid.vla.build"),
    ("SPAN_VLA_ACT", "armdroid.vla.act"),
    ("SPAN_VLA_PARSE", "armdroid.vla.parse"),
    ("SPAN_INTERACTION_SESSION_OPEN", "armdroid.interaction.session_open"),
    ("SPAN_INTERACTION_SEND_FRAME", "armdroid.interaction.send_frame"),
    ("SPAN_INTERACTION_SEND_AUDIO", "armdroid.interaction.send_audio"),
    ("SPAN_INTERACTION_EVENT", "armdroid.interaction.event"),
    ("SPAN_SAFETY_CHECK_PLAN", "armdroid.safety.check_plan"),
    ("SPAN_SAFETY_CHECK_ACTION", "armdroid.safety.check_action"),
    ("SPAN_SAFETY_DENY", "armdroid.safety.deny"),
)


def test_phase_a_span_constants_have_exact_values() -> None:
    for name, expected in _EXPECTED:
        actual = getattr(tel, name)
        assert actual == expected, f"{name}: {actual!r} != {expected!r}"


def test_phase_a_span_constants_in_all_export() -> None:
    for name, _ in _EXPECTED:
        assert name in tel.__all__, f"{name} missing from telemetry.__all__"


def test_phase_a_span_constants_are_str() -> None:
    for name, _ in _EXPECTED:
        assert isinstance(getattr(tel, name), str)


def test_legacy_span_constants_unchanged() -> None:
    """Regression: pre-existing SPAN values must NOT be perturbed."""
    assert tel.SPAN_DRIVER_CONNECT == "armdroid.driver.connect"
    assert tel.SPAN_ENV_RESET == "armdroid.env.reset"
    assert tel.SPAN_AGENT_BUILD == "armdroid.agent.build"
