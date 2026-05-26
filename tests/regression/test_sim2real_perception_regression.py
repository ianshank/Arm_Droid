"""Regression tests for Sim-to-Real and Perception PoseEstimator configuration additions.

Ensures that:
* ObjectGeometryCfg has correct fields and property helpers.
* ArmPerceptionConfig has distortion_coeffs and object_geometries fields.
* ArmSimIsaacConfig has all domain randomization fields with correct defaults.
* Range validation on ArmSimIsaacConfig (friction_range_sim, mass_scale_range)
  raises ValueError on min > max.

Marked ``regression`` to run in the dedicated stage.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from armdroid.config.schema.perception import ArmPerceptionConfig, ObjectGeometryCfg
from armdroid.config.schema.sim_isaac import ArmSimIsaacConfig

pytestmark = pytest.mark.regression


class TestObjectGeometryCfgRegression:
    """Guards ObjectGeometryCfg properties and construction."""

    def test_object_geometry_construction_and_property(self) -> None:
        cfg = ObjectGeometryCfg(keypoints_3d_m=[(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)])
        assert cfg.keypoints_3d_m == [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)]
        assert cfg.num_keypoints == 2


class TestArmPerceptionConfigRegression:
    """Guards new PoseEstimator / camera calibration parameters in ArmPerceptionConfig."""

    def test_default_perception_fields_present(self) -> None:
        cfg = ArmPerceptionConfig()
        assert cfg.distortion_coeffs == (0.0, 0.0, 0.0, 0.0)
        assert isinstance(cfg.object_geometries, dict)
        assert len(cfg.object_geometries) == 0

    def test_perception_custom_distortion_and_geometry(self) -> None:
        geom = ObjectGeometryCfg(keypoints_3d_m=[(0.0, 0.0, 0.0)])
        cfg = ArmPerceptionConfig(
            distortion_coeffs=(-0.1, 0.05, 0.0, 0.0),
            object_geometries={"test_obj": geom},
        )
        assert cfg.distortion_coeffs == (-0.1, 0.05, 0.0, 0.0)
        assert cfg.object_geometries["test_obj"].keypoints_3d_m == [(0.0, 0.0, 0.0)]


class TestArmSimIsaacConfigDomainRandomizationRegression:
    """Guards domain randomization fields and validation on ArmSimIsaacConfig."""

    def test_default_domain_randomization_fields(self) -> None:
        cfg = ArmSimIsaacConfig()
        assert cfg.randomize_physics is False
        assert cfg.friction_range_sim == (1.0, 1.0)
        assert cfg.mass_scale_range == (1.0, 1.0)
        assert cfg.stiffness_noise_pct == 0.0
        assert cfg.damping_noise_pct == 0.0

    def test_valid_custom_randomization_fields(self) -> None:
        cfg = ArmSimIsaacConfig(
            randomize_physics=True,
            friction_range_sim=(0.5, 1.5),
            mass_scale_range=(0.8, 1.2),
            stiffness_noise_pct=10.0,
            damping_noise_pct=5.0,
        )
        assert cfg.randomize_physics is True
        assert cfg.friction_range_sim == (0.5, 1.5)
        assert cfg.mass_scale_range == (0.8, 1.2)
        assert cfg.stiffness_noise_pct == 10.0
        assert cfg.damping_noise_pct == 5.0

    def test_invalid_friction_range_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ArmSimIsaacConfig(friction_range_sim=(1.5, 0.5))
        assert "friction_range_sim min" in str(exc_info.value)
        assert "swap values" in str(exc_info.value)

    def test_invalid_mass_scale_range_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ArmSimIsaacConfig(mass_scale_range=(1.2, 0.8))
        assert "mass_scale_range min" in str(exc_info.value)
        assert "swap values" in str(exc_info.value)
