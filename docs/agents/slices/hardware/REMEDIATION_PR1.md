# Remediation Plan: PR #1 — ESP32-JSON Arm Transport

> Source: peer-review findings from gemini-code-assist, devin-ai-integration,
> copilot-pull-request-reviewer, and manual static analysis of
> `feature/esp32-arm-integration`.
>
> Excluded: `_reader_loop` startup-race comment — `asyncio.create_task()` does
> not yield before `_connected = True` at the next line, so the race cannot
> fire.

**Goal**: green PR with every unresolved thread either fixed or explicitly
justified, no behavioural regressions, and documentation matching reality.

---

## Phase 1 — Safety / correctness (must merge first)

### 1. Firmware OOB write in `PumpSerial`
**File**: `firmware/arm_esp32/src/firmware.cpp` (L213-L239)

`g_rx_len` is set to `kMaxLineBytes + 1` (513) as an overflow sentinel, then
`g_rx_buf[g_rx_len] = '\0'` unconditionally writes one byte beyond the
513-byte buffer.

**Fix**: guard the null-write and dispatch on `g_rx_len <= cfg::kMaxLineBytes`:
```cpp
if (c == '\n') {
    if (g_rx_len <= cfg::kMaxLineBytes) {
        g_rx_buf[g_rx_len] = '\0';
        ProcessLine(g_rx_buf, g_rx_len);
    }
    g_rx_len = 0;
}
```

**Test**: new native target in `firmware/arm_esp32/test/` — feed a 600-byte
line followed by `\n`, assert the sentinel byte placed immediately after
`g_rx_buf` is unmodified.

---

### 2. `Esp32JsonDriver._validate_command` missing velocity check
**File**: `src/armdroid/hardware/esp32_json_driver.py` (L306-L324)

The mock driver rejects over-speed moves (`ArmCommandRejected`) but the real
driver does not, breaking the contract.

**Fix**: mirror the mock's velocity check, anchoring on:
1. Most recently *observed* state (`self._latest_state.joint_positions`) if available.
2. Else last commanded target.
3. Else `cfg.home_position`.

Raise `ArmCommandRejected` to match the contract docstring.

**Test**: parametrise `tests/contract/test_arm_driver_contract.py` so both
drivers reject an over-speed move identically.

---

### 3. `emergency_stop` not lock-serialised; asserts when disconnected
**Files**:
- `src/armdroid/hardware/esp32_json_driver.py` L203-L217 (e-stop body)
- `src/armdroid/hardware/esp32_json_driver.py` L515-L518 (`_write_blocking` assert)

**Fixes**:
- Acquire `self._send_lock` around the e-stop write (pyserial is not thread-safe).
- Replace `assert self._port is not None` with
  `raise ArmDriverError("driver not connected")`.
- Update the `emergency_stop` docstring to drop "bypasses the send lock".
- Update `_last_send_monotonic` inside the lock after the e-stop write.

**Test**: concurrent `send_joint_positions` + `emergency_stop` via fake serial —
assert the TX buffer contains only whole JSON lines (no interleaving).

---

## Phase 2 — Behavioural correctness

### 4. Primitives ignore per-primitive durations on the modern path
**File**: `src/armdroid/control/primitives.py` (L66-L160)

When `cfg.gripper_joint_index is not None`:
- `transit()`: `send_joint_positions(target, duration_s=cfg.transit_duration_s)`
- `grasp()` approach: `send_joint_positions(pre_grasp, duration_s=cfg.grasp_duration_s)`
- `place()` approach: `send_joint_positions(place_pose, duration_s=cfg.place_duration_s)`

Keep gripper open/close one-frame write at same duration.  
When `gripper_joint_index is None` keep legacy `send_joint_command` path.

**Test**: `tests/unit/control/test_primitives_gripper_joint.py` — assert
`driver._segment_duration_s` matches the configured value after each primitive.

---

### 5. Idle keepalive loop missing
**File**: `src/armdroid/hardware/esp32_json_driver.py`

Config field `transport.keepalive_interval_s` is defined and validated but the
keepalive task is never spawned.

**Fix**:
- Track `self._last_send_monotonic: float` (updated in `_send_and_await_ack`
  after write and in `emergency_stop` after the e-stop write).
- In `connect()`: spawn `_keepalive_loop` task alongside `_reader_task`.
- Loop: when `monotonic() - _last_send_monotonic >= keepalive_interval_s`,
  send a `ping` via `_send_and_await_ack`.
- In `_teardown()`: cancel and `await` the keepalive task.

HIL test at `tests/hardware/test_real_esp32_motion.py:L70-L90` must explicitly
cancel `drv._keepalive_task` after connect to keep proving firmware-side
watchdog behaviour.

**Test**: fake-serial unit test — connect, idle for `2 × keepalive_interval_s`,
assert ≥ 1 `ping` command observed on the TX queue.

---

### 6. Mock velocity check uses stale start position
**File**: `src/armdroid/hardware/mock_arm_driver.py` (L298-L309)

`_validate_command` evaluates velocity against `self._segment_target` (the
*previous* target) rather than the currently interpolated position. A chained
command issued mid-motion may pass a check it shouldn't.

**Fix**: move the velocity check into the `async with self._lock` block of
`send_joint_positions`, running it against
`current = self._interpolate_unlocked(now)`.

**Test**: while a long-duration segment is in flight, issue a second
`send_joint_positions` whose required speed from the interpolated pose exceeds
the limit but from the previous target does not — assert `ArmCommandRejected`.

---

## Phase 3 — Documentation drift

### 7. README still describes SO-ARM100 as canonical hardware
**File**: `README.md` (lines 3, 14, 41, 54, 97, 218, 313, 362, 372)

Rewrite: headline, "What is armdroid", hardware-requirements row, optional-extras
callout, directory tree, and Future-work section to reference MakerWorld 6-DoF +
gripper driven by ESP32-JSON. Cross-link
`firmware/arm_esp32/README.md` and `firmware/arm_esp32/PROTOCOL.md`.

---

### 8. Schema duration field descriptions mismatch behaviour
**File**: `src/armdroid/config/schema.py` (L334-L345)

After Phase 2 step 4 lands, tighten wording:

> "Duration over which the firmware interpolates the **approach phase** of the
> grasp/place primitive on the modern (gripper-as-joint) path. Gripper
> open/close itself is a separate one-frame write at the same duration."

---

## Phase 4 — Minor

### 9. Re-parsing JSON to recover the request id
**File**: `src/armdroid/hardware/esp32_json_driver.py` (L497-L507)

`_send_and_await_ack` re-parses the already-serialised line to get `req_id`.

**Fix**: refactor `_encode` to return `(line: bytes, req_id: str)` and consume
both in `_send_and_await_ack`.

---

## Relevant files

| File | Change |
|------|--------|
| `firmware/arm_esp32/src/firmware.cpp` | OOB fix in `PumpSerial` |
| `src/armdroid/hardware/esp32_json_driver.py` | velocity check, e-stop locking + connect-state guard, keepalive loop, `_encode` refactor |
| `src/armdroid/hardware/mock_arm_driver.py` | velocity validation under lock |
| `src/armdroid/control/primitives.py` | modern-path duration routing |
| `src/armdroid/config/schema.py` | duration field descriptions |
| `README.md` | hardware-target rewrite |
| `tests/contract/test_arm_driver_contract.py` | parametrised velocity rejection |
| `tests/unit/hardware/test_esp32_json_driver.py` | velocity, lock-serialised e-stop, disconnected e-stop, keepalive, encode refactor |
| `tests/unit/hardware/test_mock_arm_driver_modern.py` | chained mid-motion velocity test |
| `tests/unit/control/test_primitives_gripper_joint.py` | transit + approach-segment duration assertions |
| `tests/hardware/test_real_esp32_motion.py` | adjust watchdog test to cancel keepalive |
| `firmware/arm_esp32/test/test_pump_serial/` | native oversize-line test |

---

## Verification checklist

1. `ruff check src tests scripts && ruff format --check src tests scripts` — clean
2. `mypy --strict src/armdroid` — clean (matches PR baseline)
3. `pytest tests/ -q` — all 424 + new tests passing; ≤ 6 HIL skips
4. `pytest tests/contract -q` — both drivers reject identical malformed/over-speed commands
5. `pytest tests/unit/control/test_primitives_gripper_joint.py -q` — every primitive uses its named duration
6. `pytest tests/unit/hardware/test_esp32_json_driver.py::test_keepalive_pings_when_idle -q` — passes deterministically
7. `pytest tests/unit/hardware/test_esp32_json_driver.py::test_estop_serialised_with_send -q`
8. `pytest tests/unit/hardware/test_mock_arm_driver_modern.py::test_velocity_check_uses_interpolated_start -q`
9. `python scripts/gen_firmware_config.py --check` — codegen in sync
10. `pio test -e native` — new oversize-line test passes; existing interpolator tests pass
11. README rendered diff: no SO-ARM100 references in narrative copy
12. Resolve all unresolved PR #1 threads linking to the relevant commit SHA

---

## Decisions

- **Velocity-check anchor**: prefer last *observed* state over last commanded
  target — the firmware can be moving toward the previous target when the next
  command lands. Fall back to last target preserves behaviour during the brief
  window before the first heartbeat.
- **Keepalive default**: always-on. Off-by-default would re-introduce the
  watchdog footgun for callers who don't read docs.
- **HIL keepalive bypass**: test cancels `drv._keepalive_task` explicitly — no
  production toggle.
- **Commit shape**: one commit per phase so each thread can link to a specific SHA.
- **Out of scope**: reader-loop startup race (not reproducible), hash-anchored
  edit tooling, multi-agent harness (tracked under `plan-harness.md`).
