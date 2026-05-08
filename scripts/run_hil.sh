#!/usr/bin/env bash
# run_hil.sh — convenience wrapper for hardware-in-the-loop tests.
#
# Usage:
#   ./scripts/run_hil.sh [pytest-args...]
#
# Environment variables:
#   ARMDROID_HIL_RUN (default: 1)
#       Set to 0 to dry-run (tests will be collected but the hardware
#       fixture will skip them).
#   ARMDROID_HIL_PORT
#       Serial port to pass to the driver.  When set, maps to
#       ARMDROID_ARM__TRANSPORT__SERIAL_PORT so the port override is
#       picked up by Pydantic Settings without touching any config file.
#       Example: ARMDROID_HIL_PORT=/dev/ttyUSB0
#   ARMDROID_HIL_SWEEP_AMPLITUDE_RAD (default: 0.05)
#       Per-joint sweep amplitude in radians for test_per_joint_sweep.
#   ARMDROID_HIL_FAULT_INJECT (default: unset)
#       Set to 1 to include live fault-injection tests.
#
# Examples:
#   ./scripts/run_hil.sh
#   ARMDROID_HIL_PORT=/dev/ttyUSB0 ./scripts/run_hil.sh -k test_estop
#   ARMDROID_HIL_FAULT_INJECT=1 ./scripts/run_hil.sh

set -euo pipefail

export ARMDROID_HIL_RUN="${ARMDROID_HIL_RUN:-1}"

if [[ -n "${ARMDROID_HIL_PORT:-}" ]]; then
    export ARMDROID_ARM__TRANSPORT__SERIAL_PORT="${ARMDROID_HIL_PORT}"
fi

exec python -m pytest tests/hardware -m hardware -v "$@"
