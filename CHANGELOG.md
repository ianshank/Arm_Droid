# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed (PR #1 review remediation and gap analysis)

- **Named poll constants** — replaced inline magic literals in
  `Esp32JsonDriver` with `Final[float]` module-level constants:
  `_KEEPALIVE_POLL_FLOOR_S = 0.05` and `_FIRST_STATE_POLL_INTERVAL_S = 0.01`.
- **Shared fake-serial harness** — new `tests/helpers/fake_serial.py` module
  provides three canonical stand-ins (`FakeSerial`, `PingOnlyFakeSerial`,
  `SilentFakeSerial`) eliminating the three previously-duplicated private
  state-machine classes across `test_esp32_json_driver.py`,
  `test_arm_driver_contract.py`, and `test_esp32_autodetect.py`.
- **CI now runs contract tests** — `ci.yml` test step extended from
  `tests/unit tests/property` to `tests/unit tests/property tests/contract`.
- **Coverage exclusion narrowed** — `pyproject.toml` `[tool.coverage.run]
  omit` no longer blanket-excludes `esp32_json_driver.py`; truly unreachable
  paths within that file carry inline `# pragma: no cover` annotations.
- **Firmware CI step name** — `firmware-ci.yml` step renamed from
  "Run native joint-interpolator tests" to "Run native firmware unit tests
  (interpolator + serial framing)" to accurately describe the test scope.
- **Stale empty directory removed** — `testsunitscripts/` artifact at repo
  root deleted.

### Added

- **ESP32-JSON arm transport** (`Esp32JsonDriver`) — newline-delimited JSON
  over UART speaking the wire protocol in `firmware/arm_esp32/PROTOCOL.md`.
  Background reader task demuxes state/ack/nak/evt frames; per-command
  futures keyed by request id; e-stop acquires the send lock so it is
  serialised with other writes.
- **Auto-port discovery** — `cfg.arm.transport.serial_port == "auto"`
  enumerates USB serial ports via `serial.tools.list_ports`, optionally
  filtered by `usb_vid_pid_hints` and `exclude_ports`, and probes each in
  parallel for the firmware boot signature.
- **ESP32 firmware tree** under `firmware/arm_esp32/` — PlatformIO project
  (`esp32dev` Arduino target + `native` Unity test env), JSON-over-UART
  controller (50 Hz interpolator, 10 Hz heartbeat, 1 s+ host-silence
  watchdog auto-latches e-stop), wire-protocol spec, README with wiring
  diagram and troubleshooting.
- **Firmware codegen** (`scripts/gen_firmware_config.py`) — reads
  `ArmSettings` and emits `firmware/arm_esp32/src/config_generated.h`. CI
  job (`firmware-ci.yml::codegen-check`) runs `--check` to fail merges
  that change YAML without regenerating the header.
- **Structured config sub-models** — `JointLimits` (per-joint min/max +
  velocity), `ArmServoConfig` (PWM pin + pulse calibration),
  `ArmTransportConfig` (port, baud, timeouts, line cap, autodetect),
  `ArmFirmwareConfig` (interpolator hz, watchdog, version). All new
  fields have defaults; `ArmConfig` autosizes per-joint lists when `dof`
  is overridden so legacy tests still work.
- **Modern `ArmDriverProtocol` surface** — `connect`, `disconnect`,
  `is_connected`, `read_state -> ArmState`, `send_joint_positions`,
  `clear_emergency_stop`. Legacy methods preserved as backwards-compat
  adapters.
- **`ArmState`, `ArmDriverError`, `ArmCommandRejected`** value-types and
  exceptions in `armdroid.protocols`.
- **`flash_arm_esp32.sh`** — convenience wrapper around
  `pio run -t upload` with `ARMDROID_ARM__TRANSPORT__SERIAL_PORT` env-var
  port discovery.
- **`firmware-ci.yml`** — three jobs: codegen drift check, native PIO
  Unity tests for the C++ interpolator, esp32dev compile-only build.

### Changed

- **`MockArmDriver` rewritten** to implement the extended protocol with
  deterministic linear interpolation, latched e-stop, asyncio-lock-guarded
  segment state, and per-joint velocity validation. Legacy methods
  preserved (`send_joint_command` / `home` are instantaneous and clip
  silently per joint, matching the historical SO-ARM100 mock UX).
- **`build_arm_driver` real-hardware path** now constructs
  `Esp32JsonDriver` instead of `SoArm100Driver`. Mock-hardware path
  unchanged.

### Removed

- **`SoArm100Driver`** — replaced wholesale by `Esp32JsonDriver`. The
  SO-ARM100 hardware target is out; the MakerWorld ESP32 arm is in.

### Deferred to a follow-up

- **Default `dof` bump from 6 → 7** is gated on authoring 7-DoF MuJoCo
  scene XMLs (`sim/tower_of_hanoi.xml`, `sim/laundry_sorting.xml`).
  The infrastructure is in place: `ArmConfig.gripper_joint_index`
  returns ``dof - 1`` at `dof >= 7`, primitives switch to the gripper-as-joint
  code path automatically, the firmware codegen handles arbitrary joint
  counts, and the per-joint config (`joint_limits`, `servos`) auto-sizes
  when `dof` changes. Operators who want 7-DoF today can set
  `cfg.arm.dof = 7` in YAML and the primitives use the modern path; the
  default stays at 6 to keep the existing test suite green until the
  scene assets land.
- **Env-side reward shaping refresh** — `BaseArmEnv` and the per-task
  envs (`TowerOfHanoiEnv`, `LaundrySortingEnv`) read `dof` from config
  and adapt observation / action spaces dynamically, but reward shaping
  still treats the gripper as a separate concern. Folding the gripper
  joint into the reward (e.g. penalising mid-task gripper actuation)
  pairs with the dof-7 default switch.

### Documentation

- **`docs/architecture/C4.md`** — Mermaid C4 diagrams (System Context,
  Container, RL Controller component) and key design decisions.
- **README** — full setup/configuration/usage/development/roadmap rewrite.
- **Regression test layer** — `tests/regression/` baseline with
  `pytest.mark.regression`; `tests/e2e/` placeholder.
- **`.gitignore`** — `weights/`, `data/`, `datasets/`, `screenshots/`,
  `*.log`, `logs/`, `docs/_build/`, `site/` ignored.
- **Documentation** — full C4 architecture diagrams (System Context, Container,
  RL Controller component) under `docs/architecture/C4.md` with Mermaid source.
- **Regression test layer** — new `tests/regression/` directory with baseline
  regressions covering schema construction, factory DI graph, and protocol
  conformance. Marked `pytest.mark.regression`.
- **End-to-end test layer** — new `tests/e2e/` directory placeholder for full
  pipeline tests (planner → controller → driver → state read).
- **Extended `.gitignore`** — `weights/`, `data/`, `datasets/`, `screenshots/`,
  `*.log`, `logs/`, `docs/_build/`, `site/` now ignored to keep large training
  artifacts and generated docs out of git.
- **CI markers** — `regression` and `e2e` pytest markers registered in
  `pyproject.toml`.

### Changed

- **README** — replaced placeholder content with a full setup, configuration,
  usage, development, and roadmap guide. CUDA torch install via PyTorch's
  index URL is now documented; Windows MuJoCo VC++ redistributable note
  added.

## [0.1.0] — 2025-05-07

### Added

- **Initial public release.** Repository split out from
  [MouseDroidAGI](https://github.com/ianshank/MouseDroidAGI) (rover platform).
  All robot-arm code moved here; rover repo retains only the autonomous
  navigation stack.
- **Protocol-based dependency injection** — every subsystem boundary
  (`ArmDriverProtocol`, `ArmPerceptionProtocol`, `ArmPlannerProtocol`,
  `ArmControllerProtocol`, `ArmEnvironmentProtocol`) is a `@runtime_checkable`
  structural Protocol. `factory.py` is the single wiring point.
- **`ArmOrchestrator`** — lifecycle coordinator with `train()`, async
  `rollout()`, and async `shutdown()`. Drives the SAC+HER training loop
  through the controller protocol's `build_for_env()` and `train_policy()`
  methods (added during the tech-debt sweep).
- **Perception stack** — RealSense D435i depth processing, YOLO11 object
  detection (CUDA), PnP 6-DoF pose estimation, vision-to-PDDL symbolic
  state extraction, plus a unified `ArmPerception` facade.
- **Symbolic planner** — Pyperplan BFS/A* with PDDL domain + problem
  generation for Tower of Hanoi (1–10 disks, 2ⁿ−1 optimal moves) and an
  adaptive replanner with optional Anthropic Claude LLM backend.
- **RL controller** — SAC + `HerReplayBuffer` wrapper from
  Stable-Baselines3, action primitives (grasp / place / transit / home),
  cubic-interpolation trajectory generator with joint-limit clamping.
- **Environments** — `BaseArmEnv` (Gymnasium `GoalEnv`), domain randomizer,
  configurable reward shaper, curriculum manager, `TowerOfHanoiEnv`, and
  `LaundrySortingEnv`.
- **Hardware drivers** — async SO-ARM100 USB serial driver and
  in-memory `MockArmDriver` for simulation/testing.
- **Configuration system** — Pydantic v2 `ArmSettings(BaseSettings)` with
  `ARMDROID_` env prefix, `__` nested delimiter, YAML overlay loader,
  sample configs (`config/tower_of_hanoi.yaml`, `config/training.yaml`).
- **Logging** — vendored structlog setup with console / JSON renderers
  selectable via config.
- **CLI** — `python -m armdroid` and `armdroid` console-script entry
  points dispatching `train`, `rollout`, `sim` subcommands.
- **Test suite** — unit, integration, and property-based tests with
  `pytest-asyncio`, `hypothesis`, ≥85% coverage gate (currently 98.8%).
- **Tooling** — `ruff 0.8.0` lint + format, `mypy --strict`,
  `pytest-cov`, `pytest-timeout`, `pytest-xdist`.

### Changed (during the v0.1.0 lead-up)

- **`ArmControllerProtocol` extended** with `build_for_env()` and
  `train_policy()` so the orchestrator no longer reaches into
  controller internals (`controller.agent.build(env)` was a typed escape
  hatch with `# type: ignore[attr-defined]`).
- **Factory restructured** so `build_arm_orchestrator()` builds the
  hardware driver once and passes it to `build_arm_controller(cfg, driver=…)`,
  preventing two simultaneous serial connections to the SO-ARM100.
- **Imports hoisted** — `numpy`, `pathlib.Path`, and the controller-side
  `ArmEnvironmentProtocol` now live at module top instead of inside
  function bodies.

### Removed

- **Hailo accelerator path** — `arm/perception/hailo_detector.py` and the
  `HailoRuntimeProtocol` `TYPE_CHECKING` import. Hailo is Jetson-only
  hardware; armdroid runs on a desktop CUDA GPU. The `yolo_backend` config
  literal collapsed from `Literal["ultralytics", "hailo", "auto"]` to
  `Literal["ultralytics"]`.
- **Backwards-compat regression tests for legacy schemas** — armdroid
  starts at `v0.1.0`; there is no historical schema drift to defend
  against. The old `tests/regression/test_arm_config_backwards_compat.py`
  was dropped during the split.

[Unreleased]: https://github.com/ianshank/Arm_Droid/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ianshank/Arm_Droid/releases/tag/v0.1.0
