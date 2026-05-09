"""Verify env reset/step/render/close emit telemetry spans.

Closes G6 (telemetry only instrumented drivers, not envs).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from armdroid.config.schema import ArmTaskConfig, ArmTrainingConfig
from armdroid.environments.tower_of_hanoi import TowerOfHanoiEnv
from armdroid.telemetry import (
    SPAN_ENV_CLOSE,
    SPAN_ENV_RENDER,
    SPAN_ENV_RESET,
    SPAN_ENV_STEP,
    configure_telemetry,
)
from tests.helpers.recording_telemetry import RecordingTelemetry


@pytest.fixture
def recording_telemetry() -> Iterator[RecordingTelemetry]:
    """Yield a fresh recorder; restore NullTelemetry on teardown."""
    tel = RecordingTelemetry()
    configure_telemetry(tel)  # type: ignore[arg-type]
    try:
        yield tel
    finally:
        configure_telemetry(None)


def _make_env() -> TowerOfHanoiEnv:
    return TowerOfHanoiEnv(ArmTaskConfig(), ArmTrainingConfig(), dof=6)


class TestEnvSpanEmission:
    def test_reset_emits_env_reset_span(
        self,
        recording_telemetry: RecordingTelemetry,
    ) -> None:
        env = _make_env()
        env.reset(seed=42)
        assert SPAN_ENV_RESET in recording_telemetry.spans

    def test_reset_records_seed_attribute(
        self,
        recording_telemetry: RecordingTelemetry,
    ) -> None:
        env = _make_env()
        env.reset(seed=123)
        # Find the reset span and check its attrs.
        idx = recording_telemetry.spans.index(SPAN_ENV_RESET)
        assert recording_telemetry.span_attrs[idx]["seed"] == 123

    def test_step_emits_env_step_span(
        self,
        recording_telemetry: RecordingTelemetry,
    ) -> None:
        env = _make_env()
        env.reset(seed=42)
        env.step(np.zeros(6, dtype=np.float64))
        assert SPAN_ENV_STEP in recording_telemetry.spans

    def test_render_emits_env_render_span(
        self,
        recording_telemetry: RecordingTelemetry,
    ) -> None:
        env = _make_env()
        env.render()
        assert SPAN_ENV_RENDER in recording_telemetry.spans

    def test_close_emits_env_close_span(
        self,
        recording_telemetry: RecordingTelemetry,
    ) -> None:
        env = _make_env()
        env.close()
        assert SPAN_ENV_CLOSE in recording_telemetry.spans

    def test_full_episode_emits_all_span_kinds(
        self,
        recording_telemetry: RecordingTelemetry,
    ) -> None:
        env = _make_env()
        env.reset(seed=0)
        for _ in range(3):
            env.step(np.zeros(6, dtype=np.float64))
        env.render()
        env.close()
        spans = set(recording_telemetry.spans)
        assert SPAN_ENV_RESET in spans
        assert SPAN_ENV_STEP in spans
        assert SPAN_ENV_RENDER in spans
        assert SPAN_ENV_CLOSE in spans
        # Step span fired 3 times.
        assert recording_telemetry.spans.count(SPAN_ENV_STEP) == 3
