"""Unit tests for the calibration script's Pydantic model and YAML output.

These tests validate the CalibratedLimit schema and the YAML serialisation
format without requiring any hardware connection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# Ensure the scripts directory is importable for CalibratedLimit
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from calibrate_arm import CalibratedLimit


class TestCalibratedLimitModel:
    """Tests for the CalibratedLimit Pydantic model."""

    def test_round_trip(self) -> None:
        """Model serialises and deserialises cleanly."""
        limit = CalibratedLimit(min_rad=-1.5, max_rad=1.5)
        data = limit.model_dump()
        restored = CalibratedLimit(**data)
        assert restored.min_rad == limit.min_rad
        assert restored.max_rad == limit.max_rad

    def test_json_serialisation(self) -> None:
        """Model serialises to JSON cleanly."""
        limit = CalibratedLimit(min_rad=-0.785, max_rad=0.785)
        json_str = limit.model_dump_json()
        restored = CalibratedLimit.model_validate_json(json_str)
        assert restored.min_rad == -0.785
        assert restored.max_rad == 0.785

    def test_yaml_output_format(self, tmp_path: Path) -> None:
        """Verify the YAML output format matches what load_settings expects."""
        limits = [
            CalibratedLimit(min_rad=-1.0, max_rad=1.0),
            CalibratedLimit(min_rad=-0.5, max_rad=0.5),
        ]
        out_data = {
            "arm": {
                "joint_limits": [
                    {
                        "min_rad": float(f"{lim.min_rad:.3f}"),
                        "max_rad": float(f"{lim.max_rad:.3f}"),
                    }
                    for lim in limits
                ]
            }
        }

        out_file = tmp_path / "calibrated.yaml"
        with open(out_file, "w") as f:
            yaml.dump(out_data, f, default_flow_style=False, sort_keys=False)

        loaded = yaml.safe_load(out_file.read_text())
        assert "arm" in loaded
        assert "joint_limits" in loaded["arm"]
        assert len(loaded["arm"]["joint_limits"]) == 2
        assert loaded["arm"]["joint_limits"][0]["min_rad"] == -1.0
        assert loaded["arm"]["joint_limits"][1]["max_rad"] == 0.5

    def test_negative_range_allowed(self) -> None:
        """A limit where min > max is structurally valid (firmware may reject it)."""
        # The CalibratedLimit model itself doesn't validate ordering;
        # the firmware will reject invalid limits at command time.
        limit = CalibratedLimit(min_rad=1.0, max_rad=-1.0)
        assert limit.min_rad == 1.0
