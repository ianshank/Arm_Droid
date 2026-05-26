#!/usr/bin/env python3
"""Interactive calibration script for the MakerWorld 6-DoF arm.

This script allows a human operator to step each joint individually
to its physical extremes to determine safe `min_rad` and `max_rad`
limits. The results are saved to a YAML overlay file.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel

from armdroid.config.schema import ArmSettings
from armdroid.hardware.registry import get_driver

# Pydantic schema for the output YAML
class CalibratedLimit(BaseModel):
    """A single joint's calibrated angular limits."""

    min_rad: float
    max_rad: float


async def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate arm joint limits interactively.")
    parser.add_argument("--port", type=str, default="auto", help="Serial port for the ESP32")
    parser.add_argument("--output", type=Path, default=Path("config/calibrated_limits.yaml"), help="Output YAML file")
    parser.add_argument("--step-rad", type=float, default=0.05, help="Step size in radians (default: 0.05)")
    args = parser.parse_args()

    print("=== Armdroid Physical Calibration Utility ===")
    print(f"Connecting to port: {args.port}")

    # Build settings
    cfg = ArmSettings(mock_hardware=False)
    cfg.arm.transport.serial_port = args.port
    
    # Instantiate driver directly to bypass orchestrator (we only want manual control)
    factory = get_driver("esp32")
    driver = factory(cfg.arm)

    try:
        await driver.connect()
    except Exception as e:
        print(f"Failed to connect to arm: {e}")
        return 1

    print("\nConnected! Press Enter to go to the configured home position.")
    input("> ")
    try:
        await driver.home()
        # Give it time to settle
        await asyncio.sleep(2.0)
    except Exception as e:
        print(f"Failed to home arm: {e}")
        await driver.disconnect()
        return 1

    dof = driver.dof
    calibrated_limits: list[CalibratedLimit] = []
    
    current_state = await driver.read_state()
    current_pos = list(current_state.joint_positions)

    print("\n=== Calibration Instructions ===")
    print(f"For each of the {dof} joints, you will step the arm to its safe MIN and MAX limits.")
    print("Controls: 'w' (increase), 's' (decrease), 'd' (done with current limit), 'q' (quit/estop)")

    try:
        for joint_idx in range(dof):
            min_val = current_pos[joint_idx]
            for bound in ("MIN", "MAX"):
                print(f"\n--- Calibrating Joint {joint_idx} : {bound} bound ---")
                print(f"Current position: {current_pos[joint_idx]:.3f} rad")
                
                while True:
                    cmd = input(f"J{joint_idx} {bound} [w/s/d/q]: ").strip().lower()
                    if cmd == 'q':
                        print("Emergency Stop!")
                        await driver.emergency_stop()
                        return 1
                    elif cmd == 'd':
                        # Record the value
                        if bound == "MIN":
                            min_val = current_pos[joint_idx]
                        else:
                            max_val = current_pos[joint_idx]
                            calibrated_limits.append(CalibratedLimit(min_rad=min_val, max_rad=max_val))
                        break
                    elif cmd == 'w':
                        current_pos[joint_idx] += args.step_rad
                    elif cmd == 's':
                        current_pos[joint_idx] -= args.step_rad
                    else:
                        print("Invalid command.")
                        continue
                    
                    print(f"Moving Joint {joint_idx} to {current_pos[joint_idx]:.3f} rad...")
                    try:
                        await driver.send_joint_positions(tuple(current_pos), duration_s=0.5)
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        print(f"Command rejected (firmware limit?): {e}")
                        # Revert the increment
                        if cmd == 'w': current_pos[joint_idx] -= args.step_rad
                        if cmd == 's': current_pos[joint_idx] += args.step_rad
    finally:
        print("\nReturning to home position before exit...")
        try:
            await driver.home()
            await asyncio.sleep(2.0)
        except Exception:
            pass
        await driver.disconnect()

    print("\n=== Calibration Complete ===")
    out_data = {
        "arm": {
            "joint_limits": [
                {"min_rad": float(f"{limit.min_rad:.3f}"), "max_rad": float(f"{limit.max_rad:.3f}")} 
                for limit in calibrated_limits
            ]
        }
    }

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(out_data, f, default_flow_style=False, sort_keys=False)
        
    print(f"Calibration saved to {args.output}")
    print("To use this overlay, run: python -m armdroid --config config/calibrated_limits.yaml ...")

    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
