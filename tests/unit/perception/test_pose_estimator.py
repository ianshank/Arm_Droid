"""Tests for 6-DoF pose estimator."""

from __future__ import annotations

import numpy as np
import pytest

from armdroid.config.schema import ArmPerceptionConfig
from armdroid.domain.state import DetectedObject
from armdroid.perception.pose_estimator import PoseEstimator


def _make_estimator() -> PoseEstimator:
    """Create pose estimator with standard camera intrinsics."""
    cfg = ArmPerceptionConfig()
    intrinsics = np.array(
        [
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return PoseEstimator(cfg, intrinsics)


def _make_detection(cx: float = 320.0, cy: float = 240.0) -> DetectedObject:
    """Create a detection at given bounding box centre."""
    half_w = 25.0
    half_h = 25.0
    return DetectedObject(
        object_id="disk_1",
        class_name="disk_1",
        confidence=0.95,
        position_m=np.zeros(3, dtype=np.float64),
        orientation_rad=np.zeros(3, dtype=np.float64),
        bbox=np.array([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dtype=np.float64),
    )


class TestPoseEstimator:
    """Test PoseEstimator pose computation."""

    def test_centre_detection_gives_zero_xy(self) -> None:
        estimator = _make_estimator()
        det = _make_detection(cx=320.0, cy=240.0)
        depth = np.full((480, 640), 0.5, dtype=np.float32)
        pos, _ori = estimator.estimate_pose(det, depth)
        # Centre of image at (320, 240) with cx=320, cy=240 -> x=0, y=0
        assert abs(pos[0]) < 0.01
        assert abs(pos[1]) < 0.01
        assert abs(pos[2] - 0.5) < 0.01

    def test_off_centre_gives_nonzero_xy(self) -> None:
        estimator = _make_estimator()
        det = _make_detection(cx=400.0, cy=300.0)
        depth = np.full((480, 640), 0.5, dtype=np.float32)
        pos, _ = estimator.estimate_pose(det, depth)
        assert pos[0] > 0.0  # Right of centre
        assert pos[1] > 0.0  # Below centre

    def test_depth_used_for_z(self) -> None:
        estimator = _make_estimator()
        det = _make_detection()
        depth = np.full((480, 640), 1.0, dtype=np.float32)
        pos, _ = estimator.estimate_pose(det, depth)
        assert abs(pos[2] - 1.0) < 0.01

    def test_zero_depth_uses_fallback(self) -> None:
        estimator = _make_estimator()
        det = _make_detection()
        depth = np.zeros((480, 640), dtype=np.float32)
        pos, _ = estimator.estimate_pose(det, depth)
        assert pos[2] > 0.0  # Fallback depth

    def test_estimate_poses_batch(self) -> None:
        estimator = _make_estimator()
        dets = [_make_detection(cx=100.0), _make_detection(cx=500.0)]
        depth = np.full((480, 640), 0.5, dtype=np.float32)
        result = estimator.estimate_poses(dets, depth)
        assert len(result) == 2
        assert result[0].position_m[0] != result[1].position_m[0]

    def test_pnp_fallback_without_geometry(self) -> None:
        """Fallback: Orientation is zero when no geometry exists."""
        estimator = _make_estimator()
        # disk_1 is not in object_geometries
        det = _make_detection()
        depth = np.full((480, 640), 1.5, dtype=np.float32)

        pos, ori = estimator.estimate_pose(det, depth)

        assert pos.shape == (3,)
        assert ori.shape == (3,)
        np.testing.assert_array_equal(ori, np.zeros(3))

    def test_pnp_with_geometry(self) -> None:
        """Verifies that solvePnP returns a valid orientation vector for a known object."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            import pytest

            pytest.skip("cv2 not installed")

        estimator = _make_estimator()

        # Add test geometry
        from armdroid.config.schema.perception import ObjectGeometryCfg

        estimator._cfg.object_geometries["test_box"] = ObjectGeometryCfg(
            keypoints_3d_m=[
                (-0.05, -0.05, 0.0),
                (0.05, -0.05, 0.0),
                (0.05, 0.05, 0.0),
                (-0.05, 0.05, 0.0),
            ]
        )

        det = DetectedObject(
            object_id="box_1",
            class_name="test_box",
            confidence=0.95,
            position_m=np.zeros(3, dtype=np.float64),
            orientation_rad=np.zeros(3, dtype=np.float64),
            bbox=np.array([295.0, 215.0, 345.0, 265.0], dtype=np.float64),
        )

        depth = np.full((480, 640), 1.0, dtype=np.float32)

        pos, ori = estimator.estimate_pose(det, depth)

        assert pos.shape == (3,)
        assert ori.shape == (3,)

        # Given the synthetic frontal projection, rotation should be near zero.
        # Tolerance is generous because bbox corners are an approximation of the
        # true 2D-3D correspondence, not a ground-truth projection.
        np.testing.assert_allclose(ori, [0.0, 0.0, 0.0], atol=0.1)

    def test_pnp_rotated_object(self) -> None:
        """Verifies solvePnP recovers a non-zero rotation vector for an asymmetric bbox."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            import pytest

            pytest.skip("cv2 not installed")

        estimator = _make_estimator()
        from armdroid.config.schema.perception import ObjectGeometryCfg

        estimator._cfg.object_geometries["test_box"] = ObjectGeometryCfg(
            keypoints_3d_m=[
                (-0.05, -0.05, 0.0),
                (0.05, -0.05, 0.0),
                (0.05, 0.05, 0.0),
                (-0.05, 0.05, 0.0),
            ]
        )

        det = DetectedObject(
            object_id="box_2",
            class_name="test_box",
            confidence=0.90,
            position_m=np.zeros(3, dtype=np.float64),
            orientation_rad=np.zeros(3, dtype=np.float64),
            bbox=np.array([200.0, 200.0, 400.0, 300.0], dtype=np.float64),
        )

        depth = np.full((480, 640), 0.5, dtype=np.float32)
        _pos, ori = estimator.estimate_pose(det, depth)

        # Rotation vector should no longer be exactly zero
        assert not np.allclose(ori, np.zeros(3))

    def test_pnp_fallback_on_cv2_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Orientation falls back to zeros when cv2 is not importable."""
        import builtins

        from armdroid.config.schema.perception import ObjectGeometryCfg

        estimator = _make_estimator()
        estimator._cfg.object_geometries["test_box"] = ObjectGeometryCfg(
            keypoints_3d_m=[
                (-0.05, -0.05, 0.0),
                (0.05, -0.05, 0.0),
                (0.05, 0.05, 0.0),
                (-0.05, 0.05, 0.0),
            ]
        )

        det = DetectedObject(
            object_id="box_import",
            class_name="test_box",
            confidence=0.9,
            position_m=np.zeros(3, dtype=np.float64),
            orientation_rad=np.zeros(3, dtype=np.float64),
            bbox=np.array([295.0, 215.0, 345.0, 265.0], dtype=np.float64),
        )
        depth = np.full((480, 640), 1.0, dtype=np.float32)

        real_import = builtins.__import__

        def _block_cv2(name: str, *args: object, **kwargs: object) -> object:
            if name == "cv2":
                raise ImportError("cv2 blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_cv2)
        _pos, ori = estimator.estimate_pose(det, depth)
        np.testing.assert_array_equal(ori, np.zeros(3))

    def test_pnp_fallback_on_solve_exception(self) -> None:
        """Orientation falls back to zeros when solvePnP raises an exception."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            import pytest

            pytest.skip("cv2 not installed")

        from unittest.mock import patch

        from armdroid.config.schema.perception import ObjectGeometryCfg

        estimator = _make_estimator()
        estimator._cfg.object_geometries["test_box"] = ObjectGeometryCfg(
            keypoints_3d_m=[
                (-0.05, -0.05, 0.0),
                (0.05, -0.05, 0.0),
                (0.05, 0.05, 0.0),
                (-0.05, 0.05, 0.0),
            ]
        )

        det = DetectedObject(
            object_id="box_exc",
            class_name="test_box",
            confidence=0.9,
            position_m=np.zeros(3, dtype=np.float64),
            orientation_rad=np.zeros(3, dtype=np.float64),
            bbox=np.array([295.0, 215.0, 345.0, 265.0], dtype=np.float64),
        )
        depth = np.full((480, 640), 1.0, dtype=np.float32)

        with patch("cv2.solvePnP", side_effect=RuntimeError("test PnP failure")):
            _pos, ori = estimator.estimate_pose(det, depth)

        np.testing.assert_array_equal(ori, np.zeros(3))
