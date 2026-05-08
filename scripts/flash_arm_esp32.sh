#!/usr/bin/env bash
# Flash the armdroid arm controller firmware to an ESP32.
#
# Builds the PlatformIO project under firmware/arm_esp32/ and uploads
# it to the specified port. The codegen pre-build hook ensures the
# generated header is fresh before the build runs.
#
# Usage:
#   scripts/flash_arm_esp32.sh [PORT]
#
# PORT precedence:
#   1. CLI arg ($1)
#   2. ARMDROID_ARM__TRANSPORT__SERIAL_PORT env var (matches the host
#      runtime config — set this once and you don't have to remember
#      the path each time you flash)
#   3. /dev/ttyUSB0 fallback
#
# Set the port to "auto" to ask PlatformIO to pick the first ESP32 it
# can find (relies on `pio device list`).

set -euo pipefail

PORT="${1:-${ARMDROID_ARM__TRANSPORT__SERIAL_PORT:-/dev/ttyUSB0}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FW_DIR="${REPO_ROOT}/firmware/arm_esp32"

if [[ ! -d "${FW_DIR}" ]]; then
  echo "Error: firmware directory not found: ${FW_DIR}" >&2
  exit 1
fi

if ! command -v pio >/dev/null 2>&1; then
  echo "Error: PlatformIO Core not found. Install with:" >&2
  echo "  pip install --user platformio" >&2
  exit 1
fi

if [[ "${PORT}" != "auto" && ! -e "${PORT}" ]]; then
  echo "Error: serial port ${PORT} does not exist." >&2
  echo "Available USB serial ports:" >&2
  ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "  (none)" >&2
  echo
  echo "Tip: pass 'auto' to let PlatformIO pick:" >&2
  echo "  scripts/flash_arm_esp32.sh auto" >&2
  exit 1
fi

echo "[flash_arm_esp32] regenerating firmware/arm_esp32/src/config_generated.h"
python "${REPO_ROOT}/scripts/gen_firmware_config.py"

echo "[flash_arm_esp32] building firmware..."
( cd "${FW_DIR}" && pio run -e esp32dev )

echo "[flash_arm_esp32] flashing to ${PORT}..."
if [[ "${PORT}" == "auto" ]]; then
  ( cd "${FW_DIR}" && pio run -e esp32dev -t upload )
else
  ( cd "${FW_DIR}" && pio run -e esp32dev -t upload --upload-port "${PORT}" )
fi

echo "[flash_arm_esp32] done. Open the serial monitor with:"
if [[ "${PORT}" == "auto" ]]; then
  echo "  pio device monitor -b 115200"
else
  echo "  pio device monitor -p ${PORT} -b 115200"
fi
echo
echo "On a healthy boot you should see a single line like:"
echo '  {"t":"evt","kind":"boot","ver":"arm-esp32-1.0.0","ts":0.1}'
