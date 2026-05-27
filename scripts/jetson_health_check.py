#!/usr/bin/env python3
"""Jetson deployment health check for Docker HEALTHCHECK.

Probes GPU availability, armdroid package import, serial port
detection, and optionally RealSense camera connectivity.

Exit codes:
    0 — All probes passed.
    1 — One or more probes failed.

Usage:
    python scripts/jetson_health_check.py              # Full check
    python scripts/jetson_health_check.py --probe-only  # Import + GPU only
"""

from __future__ import annotations

import argparse
import sys
from typing import Final

_PROBE_GPU: Final[str] = "gpu"
_PROBE_IMPORT: Final[str] = "import"
_PROBE_SERIAL: Final[str] = "serial"
_PROBE_REALSENSE: Final[str] = "realsense"


def _check_gpu() -> tuple[bool, str]:
    """Verify CUDA GPU is available via torch."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            return True, f"GPU: {name} ({mem_mb} MB)"
        return False, "CUDA not available (torch.cuda.is_available() == False)"
    except ImportError:
        return False, "torch not installed"
    except Exception as e:
        return False, f"GPU probe error: {e}"


def _check_import() -> tuple[bool, str]:
    """Verify armdroid package imports cleanly."""
    try:
        import armdroid

        version = getattr(armdroid, "__version__", "unknown")
        return True, f"armdroid {version} imported"
    except ImportError as e:
        return False, f"armdroid import failed: {e}"
    except Exception as e:
        return False, f"armdroid import error: {e}"


def _check_serial() -> tuple[bool, str]:
    """Check for available serial ports (ESP32)."""
    try:
        from pathlib import Path

        # Check for custom udev symlink first (documented in AGENTS.md / JETSON.md)
        symlink = Path("/dev/armdroid-esp32")
        if symlink.exists():
            return True, f"Serial port found: {symlink} (custom udev symlink)"

        # Check common Jetson serial device paths
        candidates = list(Path("/dev").glob("ttyUSB*")) + list(Path("/dev").glob("ttyACM*"))
        if candidates:
            ports = ", ".join(str(p) for p in sorted(candidates))
            return True, f"Serial ports found: {ports}"
        return False, "No /dev/armdroid-esp32 or /dev/ttyUSB* or /dev/ttyACM* devices found"
    except Exception as e:
        return False, f"Serial probe error: {e}"


def _check_realsense() -> tuple[bool, str]:
    """Check RealSense camera connectivity."""
    try:
        try:
            import pyrealsense2 as rs_base
            # Verify both context and camera_info exist in the imported bindings
            if hasattr(rs_base, "context") and hasattr(rs_base, "camera_info"):
                rs = rs_base
            else:
                import pyrealsense2.pyrealsense2 as rs_sub
                rs = rs_sub
        except ImportError:
            import pyrealsense2.pyrealsense2 as rs_sub
            rs = rs_sub

        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) > 0:
            dev = devices[0]
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            return True, f"RealSense: {name} (S/N: {serial})"
        return False, "No RealSense devices detected"
    except ImportError:
        return False, "pyrealsense2 not installed (install with: pip install pyrealsense2)"
    except Exception as e:
        return False, f"RealSense probe error: {e}"


def main() -> int:
    """Run all health check probes and report status."""
    parser = argparse.ArgumentParser(description="Armdroid Jetson health check")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Run import + GPU probes only (for Docker HEALTHCHECK)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    probes: list[tuple[str, tuple[bool, str]]] = [
        (_PROBE_GPU, _check_gpu()),
        (_PROBE_IMPORT, _check_import()),
    ]

    if not args.probe_only:
        probes.append((_PROBE_SERIAL, _check_serial()))
        probes.append((_PROBE_REALSENSE, _check_realsense()))

    all_passed = True
    results: dict[str, dict[str, object]] = {}

    for name, (ok, msg) in probes:
        results[name] = {"ok": ok, "message": msg}
        if not ok:
            all_passed = False

        if not args.json:
            icon = "OK" if ok else "FAIL"
            print(f"  [{icon}] {name}: {msg}")

    if args.json:
        import json

        print(json.dumps({"healthy": all_passed, "probes": results}, indent=2))

    if not args.json:
        print()
        if all_passed:
            print("Health check: ALL PASSED")
        else:
            print("Health check: FAILED")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
