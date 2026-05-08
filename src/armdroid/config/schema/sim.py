"""MuJoCo simulation configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ArmSimConfig(BaseModel):
    """MuJoCo simulation configuration for robot arm training."""

    scene_path: Path = Field(
        Path("sim/tower_of_hanoi.xml"),
        description="MuJoCo scene XML path",
    )
    timestep_s: float = Field(0.002, gt=0, description="Physics timestep (s)")
    n_substeps: int = Field(5, gt=0, description="Physics substeps per step")
    render_width: int = Field(640, gt=0, description="Render width (px)")
    render_height: int = Field(480, gt=0, description="Render height (px)")
    domain_randomization: bool = Field(True, description="Enable domain randomization")
    mass_range_pct: float = Field(20.0, ge=0, le=100, description="Mass variation range (%)")
    friction_range: float = Field(0.3, ge=0, description="Friction coefficient variation")
    position_noise_m: float = Field(0.005, ge=0, description="Object position noise (m)")
    lighting_variation: float = Field(0.2, ge=0, le=1, description="Lighting intensity variation")
    camera_pose_noise_deg: float = Field(10.0, ge=0, description="Camera pose noise (degrees)")


__all__ = ["ArmSimConfig"]
