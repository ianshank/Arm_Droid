"""Perception stack configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ObjectGeometryCfg(BaseModel):
    """3D geometry keypoints for known objects (used by PnP)."""

    keypoints_3d_m: list[tuple[float, float, float]] = Field(
        ...,
        description="List of 3D keypoints in the object's local coordinate frame (m).",
    )

    @property
    def num_keypoints(self) -> int:
        """Return the number of 3D keypoints defined for this object."""
        return len(self.keypoints_3d_m)


class ArmPerceptionConfig(BaseModel):
    """Perception stack configuration for robot arm platform.

    Note: ``yolo_backend`` is fixed to ``"ultralytics"`` in armdroid (the
    Hailo accelerator path was Jetson-specific and has been removed).
    The field is preserved for forward compatibility if other backends
    (e.g. TensorRT) are added later.
    """

    depth_camera_type: Literal["realsense_d435i", "oak_d", "zed2i", "mock"] = Field(
        "realsense_d435i",
        description="Depth camera hardware type",
    )
    yolo_model_path: Path = Field(
        Path("models/yolo11_disk_detector.pt"),
        description="YOLO model weights path",
    )
    yolo_confidence_threshold: float = Field(
        0.5, gt=0, le=1, description="YOLO detection confidence threshold"
    )
    yolo_nms_iou_threshold: float = Field(
        0.45,
        gt=0,
        le=1,
        description="YOLO NMS IoU threshold for non-maximum suppression",
    )
    yolo_backend: Literal["ultralytics"] = Field(
        "ultralytics",
        description="YOLO inference backend (ultralytics on CUDA)",
    )
    pose_estimator: Literal["pnp", "learned"] = Field(
        "pnp",
        description="Pose estimation method",
    )
    pose_tolerance_m: float = Field(0.005, gt=0, description="Pose estimation tolerance (m)")
    detection_fps: float = Field(30.0, gt=0, description="Detection rate (Hz)")
    depth_min_m: float = Field(0.01, gt=0, description="Minimum valid depth (m)")
    depth_max_m: float = Field(10.0, gt=0, description="Maximum valid depth (m)")
    depth_hole_threshold_m: float = Field(
        0.02, gt=0, description="Depth below which pixels are treated as holes (m)"
    )
    depth_filter_kernel_size: int = Field(
        3, gt=0, description="Median filter kernel size for depth noise reduction"
    )
    fallback_depth_m: float = Field(
        0.3, gt=0, description="Fallback depth when centre pixel is invalid (m)"
    )
    invalid_depth_threshold_m: float = Field(
        0.01, ge=0, description="Depth values below this are considered invalid (m)"
    )
    white_brightness_threshold: float = Field(
        200.0, ge=0, le=255, description="Brightness above which garment is classified white"
    )
    white_saturation_threshold: float = Field(
        0.15, ge=0, le=1, description="Saturation below which bright garment is white"
    )
    dark_brightness_threshold: float = Field(
        80.0, ge=0, le=255, description="Brightness below which garment is classified dark"
    )
    default_focal_length: float = Field(500.0, gt=0, description="Default camera focal length (px)")
    default_principal_x: float = Field(320.0, gt=0, description="Default principal point X (px)")
    default_principal_y: float = Field(240.0, gt=0, description="Default principal point Y (px)")
    distortion_coeffs: tuple[float, ...] = Field(
        default=(0.0, 0.0, 0.0, 0.0),
        description=(
            "Camera lens distortion coefficients (k1, k2, p1, p2, ...). "
            "Passed directly to cv2.solvePnP. Default assumes no distortion."
        ),
    )
    object_geometries: dict[str, ObjectGeometryCfg] = Field(
        default_factory=dict,
        description="Mapping from object class name to its 3D geometry keypoints.",
    )


__all__ = ["ArmPerceptionConfig", "ObjectGeometryCfg"]
