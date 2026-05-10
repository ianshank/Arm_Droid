# armdroid ESP32 Firmware

Arm controller firmware for the MakerWorld 6-DoF + gripper arm, driven
by an ESP32 DevKitC over USB UART. Speaks the wire protocol in
[`PROTOCOL.md`](PROTOCOL.md) — newline-delimited JSON at 115 200 baud —
to the Python host driver in
`src/armdroid/hardware/esp32_json_driver.py` (added in commit 6 of the
ESP32 integration).

## Build & flash

```bash
# Install PlatformIO Core (one-time)
pip install --user platformio

# Build only
pio run -e esp32dev

# Build + flash + open serial monitor
pio run -e esp32dev -t upload --upload-port /dev/ttyUSB0
pio device monitor -p /dev/ttyUSB0 -b 115200

# Convenience wrapper from the repo root
bash scripts/flash_arm_esp32.sh /dev/ttyUSB0
```

On a healthy boot you should see, within ~100 ms of the port opening:

```
{"t":"evt","kind":"boot","ver":"arm-esp32-1.0.0","ts":0.1}
```

After that, asynchronous `state` heartbeats arrive at the configured
`heartbeat_hz` (default 10 Hz).

## Native unit tests

```bash
pio test -e native
```

Runs the C++ joint-interpolator tests under Unity on the host without
flashing hardware. Useful to verify changes to `src/interpolator.h`
before reaching for a board.

## Configuration source of truth

The header `src/config_generated.h` is **auto-generated** by
`scripts/gen_firmware_config.py` from the host-side
`armdroid.config.schema.ArmSettings`. Do **not** edit it by hand. To
change joint limits, servo pulse calibration, pin map, watchdog timeout,
or wire framing constants, edit the YAML under `config/` (or the
defaults in `src/armdroid/config/schema.py`) and regenerate:

```bash
python scripts/gen_firmware_config.py
```

The PlatformIO `extra_scripts = pre:gen_config_pre.py` line
runs the regenerator automatically before every `pio run`. CI also runs
`scripts/gen_firmware_config.py --check` to fail merges that change the
YAML without committing the regenerated header.

## Wiring

| Joint           | Servo type | ESP32 GPIO (default) |
|-----------------|------------|----------------------|
| 0 base_yaw      | MG995      | 13                   |
| 1 shoulder_pitch| MG995      | 14                   |
| 2 elbow_pitch   | MG995      | 27                   |
| 3 wrist_pitch   | MG995      | 26                   |
| 4 wrist_roll    | MG90S      | 25                   |
| 5 wrist_yaw     | MG90S      | 33                   |
| 6 gripper       | MG90S      | 32                   |

Pin map and pulse calibration are configurable in
`src/armdroid/config/schema.py` (`ArmServoConfig`). Avoid ESP32 strapping
pins (0, 2, 5, 12, 15) and input-only pins (34–39).

Construction-time wiring, power-distribution, and joint stack-up
diagrams live in [`docs/architecture/WIRING.md`](../../docs/architecture/WIRING.md).

Power: each MG995 can draw ~1 A peak under load — do **not** power the
servos from the ESP32's 5 V regulator. Use a separate 5 V / 5 A supply
common-grounded with the ESP32.

## E-stop watchdog

If no command arrives for `cfg.arm.firmware.watchdog_timeout_s` (default
2 s), the firmware auto-latches e-stop and emits
`{"t":"evt","kind":"fault","code":"estop","msg":"watchdog or host request"}`.
The host driver issues a periodic `ping` (every
`cfg.arm.transport.keepalive_interval_s`, default 0.5 s) to keep the
watchdog satisfied between motion commands.

The latch survives until the host sends `clear_estop`. While latched,
all `set_joints` commands are NAK'd with `err: "estop_latched"`.

## Troubleshooting

| Symptom                                | Likely cause                                            |
|----------------------------------------|---------------------------------------------------------|
| No `boot` event after flashing         | Wrong board variant; try `pio run -e esp32dev -v` to see toolchain output |
| `out_of_range` NAKs for valid commands | YAML changed but `config_generated.h` not regenerated   |
| Watchdog latches mid-task              | Increase `cfg.arm.firmware.watchdog_timeout_s` or decrease `keepalive_interval_s` |
| Servos buzz / chatter                  | Servo PSU drooping under load; check current capacity   |
| Random reboots                         | USB undervolt; use a USB 3 port or powered hub          |
