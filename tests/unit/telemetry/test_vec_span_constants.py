"""Vec-path SPAN_* constants are exported with stable string identifiers."""

from __future__ import annotations

from armdroid import telemetry

_EXPECTED: dict[str, str] = {
    "SPAN_AGENT_BUILD_VEC": "armdroid.agent.build_vec",
    "SPAN_AGENT_TRAIN_VEC": "armdroid.agent.train_vec",
    "SPAN_ENV_VEC_RESET": "armdroid.env.vec_reset",
    "SPAN_ENV_VEC_STEP": "armdroid.env.vec_step",
    "SPAN_ENV_VEC_CLOSE": "armdroid.env.vec_close",
    "SPAN_ENV_VEC_KIT_BOOT": "armdroid.env.vec_kit_boot",
}


def test_vec_span_constants_present() -> None:
    """Each vec-path SPAN_* constant exists with its documented value."""
    for name, value in _EXPECTED.items():
        assert hasattr(telemetry, name), f"missing {name}"
        assert getattr(telemetry, name) == value


def test_vec_span_constants_in_all() -> None:
    """All vec SPAN_* constants are listed in telemetry.__all__."""
    for name in _EXPECTED:
        assert name in telemetry.__all__, f"{name} missing from __all__"
