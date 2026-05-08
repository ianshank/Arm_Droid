# Arm Wire Protocol v1

Newline-delimited JSON over UART, default 115 200 baud 8N1.

The host (Python `armdroid.hardware.esp32_json_driver.Esp32JsonDriver`)
opens the serial port and exchanges single-line JSON documents with the
ESP32 firmware. Every line is a complete JSON object terminated by `\n`.
The firmware never emits partial frames; the host must tolerate arbitrary
line ordering between command replies and heartbeats.

## Common envelope fields

Every message has `t` (message type) and almost always `ts`. The other
envelope fields are present only on the message types listed below — host
implementations MUST tolerate their absence on other frame types.

| field | type   | present on              | meaning                              |
|-------|--------|-------------------------|--------------------------------------|
| `t`   | string | all                     | message type                         |
| `id`  | int    | `cmd`, `ack`, `nak`     | monotonic request id (host→fw)       |
| `seq` | int    | `state`                 | monotonic sequence (fw→host)         |
| `ts`  | float  | all except `evt:boot/fault` envelopes that omit it | sender's monotonic timestamp in seconds |

In practice: `state` frames carry `seq` + `ts` but no `id`; `evt` frames
carry `kind` + `ts` (no `id`, no `seq`); `ack`/`nak` carry `id` + `ts`
(no `seq`).

## Host → firmware (`t = "cmd"`)

| `cmd` value   | payload field                | description                           |
|---------------|------------------------------|---------------------------------------|
| `set_joints`  | `q` (list[N]), `dur_ms` (int)| Move to joint vector over duration    |
| `estop`       | —                            | Latch emergency stop                  |
| `clear_estop` | —                            | Release latch                         |
| `ping`        | —                            | Liveness check                        |
| `get_state`   | —                            | Force an immediate state frame        |

`q` length must equal the firmware's compile-time `kNumJoints` (sourced
from `ArmConfig.dof` via `scripts/gen_firmware_config.py`). Joints are
radians except the gripper joint (last index when present), which is
normalised `[0, 1]` with `0` = open and `1` = closed.

The firmware applies its own joint limits in addition to whatever the
host sent — limit checks happen on both sides defence-in-depth.

## Firmware → host

### `t = "ack"`
Sent in direct reply to every successful `cmd`. Echoes `id`. No payload.

### `t = "nak"`
Sent in reply to a rejected `cmd` whose request id is recoverable.
Fields: `id`, `err` (string code), `msg` (human-readable). Error codes:

| code              | meaning                                          |
|-------------------|--------------------------------------------------|
| `bad_shape`       | Required field missing or wrong type             |
| `bad_joint_count` | `q` had wrong number of entries                  |
| `out_of_range`    | A joint value violated firmware-side limits      |
| `estop_latched`   | Motion command issued while e-stop active        |
| `unknown_cmd`     | `cmd` value not recognised                       |

Malformed JSON cannot be reported as `nak` because the request `id` is
not recoverable — the firmware emits an `evt` of `kind = "fault"` with
`code = "bad_json"` instead. See the `evt` section below.

### `t = "state"`
Asynchronous heartbeat at `cfg.transport.heartbeat_hz`. Fields:

* `q`  — current joint positions (length N)
* `qd` — current joint velocities, rad/s (length N; zeros open-loop)
* `mv` — bool, true if the firmware is interpolating
* `es` — bool, true if e-stop is latched
* `seq`, `ts`

### `t = "evt"`
Out-of-band event. `kind` field. Currently defined:

* `boot`  — sent once on startup with `ver` (firmware version string)
* `fault` — fields `code`, `msg` (e.g. servo write timeout)

## Framing rules

* Host MUST drain stale lines on connect by sending `ping` and waiting
  for the matching `ack` before issuing motion commands. The number of
  drain pings is configurable via
  `cfg.arm.transport.drain_pings_on_connect`.
* Firmware MUST send a `boot` event within `cfg.arm.transport.connect_timeout_s`
  of the host opening the port.
* Lines longer than `cfg.arm.transport.max_line_bytes` (default 512) are
  invalid; firmware drops them silently.

## Why JSON and not a binary protocol

The orchestrator runs at 30 Hz, command frames are ~120 bytes, and the
ESP32's hardware UART handles 115 200 baud with margin. Binary framing
would shave ~50 % off the bandwidth but cost debuggability — being able
to `cat /dev/ttyUSB0` and read the conversation is worth it for a hobby
testbed. Migrate to MessagePack or COBS only if the 30 Hz budget shows
measurable pressure under profiling.
