// Firmware compile-time configuration shim.
//
// All actual values live in config_generated.h, which is produced from
// armdroid.config.schema.ArmSettings by scripts/gen_firmware_config.py
// at host-side build time. This shim is the only #include the rest of
// the firmware sees so the codegen pipeline is invisible to user code.
//
// To regenerate after editing the YAML config:
//
//     python scripts/gen_firmware_config.py
//
// CI runs `--check` to fail the build if the on-disk header drifts.

#pragma once

#include "config_generated.h"
