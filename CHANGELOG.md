# Changelog

<!-- markdownlint-disable MD024 -->

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **ESP32 MakerWorld platform support** — newline-delimited JSON-over-UART
  transport, auto-port discovery, PlatformIO firmware tree, firmware
  codegen (`scripts/gen_firmware_config.py`), and a dedicated
  `firmware-ci.yml` workflow for codegen drift, native tests, and ESP32
  builds.
- **Structured hardware + config primitives** — richer `ArmConfig`
  sub-models (`JointLimits`, servo calibration, transport, firmware), a
  modern `ArmDriverProtocol` (`connect`, `disconnect`, `read_state`,
  `send_joint_positions`, `clear_emergency_stop`), and the supporting
  `ArmState` / `ArmDriverError` / `ArmCommandRejected` domain types.
- **Enterprise foundation (v0.2)** — stable `armdroid.api` public surface
  re-exported from `armdroid`, pure `armdroid.domain` modules, typed
  `Registry[T]` infrastructure, subsystem registries + PEP 621 entry-point
  groups, a migration guide, and ADRs documenting the layering,
  registry-pattern, and shim strategy.
- **Compatibility bridge** — `armdroid.compat.LegacyArmDriverAdapter` plus
  shim modules for moved imports such as `armdroid.protocols`,
  `armdroid.factory`, `armdroid.orchestrator`, and
  `armdroid.hardware.esp32_json_driver`, all emitting
  `DeprecationWarning` while preserving legacy callers through the
  migration window.
- **Developer workflow hardening** — `Makefile`, `.pre-commit-config.yaml`,
  refreshed GitHub Actions matrix, Dependabot configuration, shared
  fake-serial test helpers, and regression coverage around the new public
  surface.

### Fixed

- **CI gate stability** — Windows test job no longer fails to parse
  multi-line `pytest` invocations under PowerShell (collapsed to a single
  line). The ESP32 protocol-conformance regression test now skips cleanly
  when the optional `[hardware]` extra (pyserial) is absent instead of
  asserting against an environment that legitimately cannot construct the
  driver. `firmware-ci/codegen-check` and `pio-build` now install
  `numpy>=1.24` because the public API re-exports protocol types whose
  annotations reference `numpy.typing.NDArray`.
- **Plugin entry-point hygiene** — the `armdroid.drivers` group now points
  the `esp32` entry at the canonical `armdroid.hardware.esp32` package
  instead of the deprecated shim, so `importlib.metadata.entry_points`
  no longer triggers a spurious `DeprecationWarning`.
- **Domain protocol typing** — `numpy` / `numpy.typing` are now imported
  under `TYPE_CHECKING` in `armdroid.domain.protocols`, keeping the
  protocol surface pure and preventing optional-dependency import failures
  in lightweight tooling environments (e.g. firmware codegen).
- **Documentation polish** — restored clickable status badges in
  `README.md`, fixed the broken ADR cross-reference in
  `docs/architecture/PHASES.md` (now points at ADR-0003), and switched
  the migration guide's `sed -i` example to portable `sed -i.bak` syntax.
- **Dependabot ergonomics** — added explicit `timezone: "Etc/UTC"` to both
  the `pip` and `github-actions` schedule blocks for unambiguous timing.

### Changed

- **`MockArmDriver`** now implements the extended protocol with
  deterministic interpolation, latched e-stop behaviour, lock-guarded
  segment state, and legacy method semantics preserved for existing
  callers.
- **Package layout** moved from the flat v0.1 tree to explicit layered
  packages: `api`, `domain`, `config`, `orchestration`, `compat`, and
  subsystem modules with canonical import paths.
- **ESP32 driver implementation** was decomposed into
  `armdroid.hardware.esp32.driver`, `framing`, `portfinder`, and
  `validator`; `esp32_json_driver.py` remains as a compatibility shim.
- **Composition wiring** now resolves drivers and environments through
  typed factory aliases from registries, removing the remaining
  `# type: ignore[call-arg]` escapes from factory dispatch.
- **Regression and documentation surfaces** now track the v0.2 layout:
  refreshed C4 diagrams, README, migration guide, and regression tests for
  public API and compatibility guarantees.

### Fixed

- **Named polling constants** replaced inline timing literals in the ESP32
  driver so polling behaviour is explicit and regression-testable.
- **Coverage and logging gap analysis** added explicit tests for ESP32
  driver error paths, port autodetect edge cases, PDDL normalisation,
  symbolic-planner fallback, perception pose-estimation dispatch,
  structured logger naming, and Anthropic replanner retry/cache branches.
- **Pre-PR validation gate** now passes cleanly with `ruff`,
  `mypy --strict`, and `pytest --cov`, validating this branch at
  530 passing tests, 6 skipped, and 98% total coverage.

### Removed

- **`SoArm100Driver`** — replaced wholesale by `Esp32JsonDriver`. The
  SO-ARM100 hardware target is out; the MakerWorld ESP32 arm is in.
- **Monolithic ESP32 driver as the canonical edit surface** — new driver
  work should target the split `armdroid.hardware.esp32` package instead
  of the legacy shim module.

### Deferred to a follow-up

- **Default `dof` bump from 6 → 7** is still gated on proper MuJoCo scene
  assets and sim-to-real validation. Operators can opt in via YAML today;
  the default stays at 6 until the scene and regression story are ready.
- **Env-side reward-shaping refresh** remains paired with the eventual
  7-DoF default switch so gripper-as-joint behaviour is reflected in the
  reward model as well as the control path.

### Documentation

- **Architecture docs** — `docs/architecture/C4.md` now models the stable
  public API, layered package layout, registry seams, and the v0.2
  composition root.
- **README** — refreshed to match the v0.2 package layout, local developer
  workflow (`make`, `pre-commit`, CI), and post-foundation roadmap.
- **Regression layer** — `tests/regression/` now guards compatibility,
  public-surface imports, and baseline behavioural contracts alongside the
  broader unit/property/contract suites.
- **Ignore rules** — `.gitignore` now covers local hook cache and the large
  generated artifacts associated with training, docs, logs, and firmware
  builds.

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
