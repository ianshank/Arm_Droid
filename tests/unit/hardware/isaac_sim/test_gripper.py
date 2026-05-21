"""Unit tests for gripper conversion (PR-B B.6).

Closes R5 + R7 from PR #8 review. Pure-Python tests — do NOT import
isaaclab. Verifies:
- The boundary cases ``normalised=0`` ↔ ``radians_open`` and
  ``normalised=1`` ↔ ``radians_closed``.
- Round-trip identity over a Hypothesis-generated [0, 1] sample.
- Degenerate config raises ``ValueError`` (catches both equality
  cases via ``math.isclose``).
- Vector helpers preserve non-gripper joints unchanged.
- Out-of-range gripper indices raise ``IndexError``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from armdroid.config.schema.sim_isaac import ArmSimIsaacConfig
from armdroid.hardware.isaac_sim.gripper import (
    normalised_to_radians,
    normalised_vector_to_radians,
    radians_to_normalised,
    radians_vector_to_normalised,
)


@pytest.fixture
def cfg() -> ArmSimIsaacConfig:
    return ArmSimIsaacConfig()


class TestBoundaryCases:
    def test_normalised_zero_maps_to_radians_open(self, cfg: ArmSimIsaacConfig) -> None:
        out = normalised_to_radians(cfg.gripper_normalised_open, cfg)
        assert math.isclose(out, cfg.gripper_joint_radians_open, abs_tol=1e-12)

    def test_normalised_one_maps_to_radians_closed(self, cfg: ArmSimIsaacConfig) -> None:
        out = normalised_to_radians(cfg.gripper_normalised_closed, cfg)
        assert math.isclose(out, cfg.gripper_joint_radians_closed, abs_tol=1e-12)

    def test_radians_open_maps_to_normalised_zero(self, cfg: ArmSimIsaacConfig) -> None:
        out = radians_to_normalised(cfg.gripper_joint_radians_open, cfg)
        assert math.isclose(out, cfg.gripper_normalised_open, abs_tol=1e-12)

    def test_radians_closed_maps_to_normalised_one(self, cfg: ArmSimIsaacConfig) -> None:
        out = radians_to_normalised(cfg.gripper_joint_radians_closed, cfg)
        assert math.isclose(out, cfg.gripper_normalised_closed, abs_tol=1e-12)


class TestRoundTrip:
    def test_round_trip_over_linspace(self, cfg: ArmSimIsaacConfig) -> None:
        for x in np.linspace(0, 1, 11):
            rt = radians_to_normalised(normalised_to_radians(float(x), cfg), cfg)
            assert math.isclose(
                rt, float(x), abs_tol=1e-9
            ), f"round-trip lost precision at x={x}: rt={rt}"

    @given(x=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_round_trip_hypothesis(self, cfg: ArmSimIsaacConfig, x: float) -> None:
        # ``cfg`` is read-only here; safe to share across Hypothesis examples.
        rt = radians_to_normalised(normalised_to_radians(x, cfg), cfg)
        assert math.isclose(rt, x, abs_tol=1e-9)


class TestDegenerateConfig:
    def test_radians_open_equal_closed_raises(self, cfg: ArmSimIsaacConfig) -> None:
        bad = cfg.model_copy(
            update={
                "gripper_joint_radians_open": 0.5,
                "gripper_joint_radians_closed": 0.5,
            }
        )
        with pytest.raises(ValueError, match="gripper_joint_radians"):
            normalised_to_radians(0.0, bad)

    def test_normalised_open_equal_closed_raises(self, cfg: ArmSimIsaacConfig) -> None:
        bad = cfg.model_copy(
            update={
                "gripper_normalised_open": 0.5,
                "gripper_normalised_closed": 0.5,
            }
        )
        with pytest.raises(ValueError, match="gripper_normalised"):
            normalised_to_radians(0.0, bad)

    def test_isclose_catches_drift_within_rel_tol(self, cfg: ArmSimIsaacConfig) -> None:
        """Tiny drift below rel_tol still treated as degenerate (intentional)."""
        bad = cfg.model_copy(
            update={
                "gripper_joint_radians_open": 0.5,
                "gripper_joint_radians_closed": 0.5 + 1e-15,
            }
        )
        with pytest.raises(ValueError, match="degenerate"):
            normalised_to_radians(0.0, bad)


class TestVectorHelpers:
    def test_normalised_vector_converts_only_gripper(self, cfg: ArmSimIsaacConfig) -> None:
        # Other joints stay untouched (in radians); gripper element becomes
        # gripper_joint_radians_open since input value 0.0 is "open".
        joints = (0.1, 0.2, 0.3, 0.4, 0.5, 0.0)
        out = normalised_vector_to_radians(joints, gripper_index=5, cfg=cfg)
        # Non-gripper joints unchanged
        assert out[:5] == joints[:5]
        # Gripper element converted
        assert math.isclose(out[5], cfg.gripper_joint_radians_open, abs_tol=1e-12)

    def test_radians_vector_round_trip(self, cfg: ArmSimIsaacConfig) -> None:
        joints_norm = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7)
        joints_rad = normalised_vector_to_radians(joints_norm, 5, cfg)
        rt = radians_vector_to_normalised(joints_rad, 5, cfg)
        for original, recovered in zip(joints_norm, rt, strict=True):
            assert math.isclose(original, recovered, abs_tol=1e-9)

    def test_out_of_range_gripper_index_raises(self, cfg: ArmSimIsaacConfig) -> None:
        joints = (0.0, 0.0, 0.0)
        with pytest.raises(IndexError, match="out of range"):
            normalised_vector_to_radians(joints, gripper_index=5, cfg=cfg)

    def test_negative_gripper_index_raises(self, cfg: ArmSimIsaacConfig) -> None:
        joints = (0.0, 0.0, 0.0)
        with pytest.raises(IndexError, match="out of range"):
            normalised_vector_to_radians(joints, gripper_index=-1, cfg=cfg)
