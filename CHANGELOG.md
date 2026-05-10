# Changelog

<!-- markdownlint-disable MD024 -->

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Isaac Sim / Isaac Lab runtime integration (PR-B, opt-in `[isaac]` extra)** —
  the actual NVIDIA Isaac Sim 5.1 / Isaac Lab 2.3 runtime code, behind
  the `armdroid[isaac]` extra. Default install (`pip install -e ".[dev]"`)
  is unchanged. ADR-0005 records the architectural decisions.
  - **`armdroid.config.schema.sim_isaac.ArmSimIsaacConfig`** — 40+ Pydantic
    fields covering every Isaac Sim numeric / string knob (PD gains, init
    pose, env id, joint names, gripper conversion). No hardcoded values
    in downstream consumers. Also exports `_default_sim_cfg()` factory.
  - **`armdroid.config.schema.training.RslRlPpoConfig`** — RSL-RL PPO
    hyperparameters as Pydantic fields with bounds.
  - **`ArmTrainingConfig.algorithm`** Literal extended to include
    `"rsl_rl_ppo"`. **`ArmTaskConfig.task_type`** Literal extended to
    include `"so_arm_reach_isaac"`. **`arm_driver_kind`** Literal +
    `DriverKind` widened to include `"isaac_sim"`.
  - **`armdroid.hardware.isaac_sim`** — `IsaacSimDriver` (full
    `ArmDriverProtocol` impl using Isaac Lab 2.3's singular
    `set_joint_position_target` API; lazy isaaclab/torch imports;
    AppLauncher process-singleton guard), `gripper.py` (pure
    conversion functions, single source of truth for the
    URDF-radians ↔ armdroid-normalised rescale-and-invert),
    `articulation.py` (vendored `ArticulationCfg` builder, every PD
    gain parametrised on `ArmSimIsaacConfig`).
  - **`armdroid.environments.isaac`** — `SoArmReachIsaacEnv`
    (`ArmEnvironmentProtocol` wrapper around Isaac Lab
    `ManagerBasedRLEnv`); `_TensorAdapter` at
    `armdroid.environments._tensor_adapter` (defensive `.reshape(-1)[0]`
    for Isaac Lab's `(num_envs,)` reward shape; raises on `num_envs > 1`).
    Vendored `tasks/reach/*` from MuammerBay/isaac_so_arm101 (BSD-3,
    pinned `e4624dea075b00a36dbc66bebd531d191c92e8cd`).
  - **`armdroid.control.rsl_rl_agent.RslRlPpoAgent`** — RSL-RL
    `OnPolicyRunner` wrapper conforming to `ArmRLAgentProtocol`.
    Lazy `from rsl_rl.runners import OnPolicyRunner` inside
    `build()`. `predict()` does numpy↔torch conversion at the
    protocol boundary. Reaches through `env._isaac_env` for the raw
    vec-env (RSL-RL requires torch tensors, not numpy scalars).
  - **`build_arm_controller` branch** — `algorithm == "rsl_rl_ppo"`
    instantiates `RslRlPpoAgent(ppo_cfg=cfg.arm_rsl_rl_ppo,
    training_cfg=cfg.arm_training)` directly rather than via the
    registry's `Callable[..., ArmRLAgentProtocol]` factory; avoids
    dropping YAML overlays via `ArmSettings()` re-read (peer-review C-1).
  - **NVIDIA index URL** — README + `gpu-ci.yml` document
    `pip install -e ".[isaac]" --extra-index-url https://pypi.nvidia.com`.
    `isaaclab[all,isaacsim]==2.3.0` is hosted on `pypi.nvidia.com`,
    not PyPI. NO torch/torchvision pin in `[isaac]` (R2-safe).
  - **`tests/isaac/`** smoke suite — `tests/isaac/conftest.py` mirrors
    `tests/hardware/conftest.py`'s env-var-gated pattern
    (`ARMDROID_ISAAC_RUN=1`); auto-marks every test `@pytest.mark.isaac`
    + `@pytest.mark.gpu`. `test_smoke_isaac_driver.py` covers the full
    driver lifecycle; `test_smoke_isaac_env.py` covers reset → step×5
    → close. Local-only (CI doesn't install isaaclab).
  - **`.github/workflows/gpu-ci.yml`** — manual-dispatch + `gpu` PR
    label + Isaac path-trigger. `ubuntu-latest` runner installs
    `[dev,isaac]` (~8-10 GB), asserts the import surface, runs the
    pure-Python isaac-extra unit tests. The actual GPU-bound smoke
    suite runs locally only.
  - **`config/tower_of_hanoi_isaac.yaml`** — overlay flipping the path
    from MuJoCo + ESP32 to Isaac Sim + RSL-RL PPO.
  - **`tests/property/test_arm_env_invariants.py`** — extended
    `_BUILTIN_ENVS` with `"so_arm_reach_isaac"` gated on
    `importlib.util.find_spec("isaaclab")` (no import side-effects).
  - **THIRD_PARTY_NOTICES.md** — indexed MuammerBay vendored content
    alongside the existing TheRobotStudio entry, with import-rewrite
    documentation.
  - **ADR-0005** — `docs/architecture/ADR/ADR-0005-isaac-sim-backend.md`
    records 10 decisions (extras pattern, singular API, _TensorAdapter
    `num_envs == 1`, gripper convention preserved, RslRlPpoAgent
    factory branch, vendored BSD-3 attribution, AppLauncher singleton,
    no torch pin, USD as build artefact, GPU CI manual-dispatch).

- **Isaac Sim / Isaac Lab integration scaffolding (PR-A, no runtime dep)** —
  groundwork for an opt-in NVIDIA Isaac Sim 5.1 / Isaac Lab 2.3 backend
  behind a future `armdroid[isaac]` extra. Closes 4 reviewer-flagged
  code-side defects from PR #8 review (R1, R2, R3, R6 — the
  doc-only R4 / R5 / R7 / R8 are moot after PR #9 reverted the
  research doc on main) and 7 pre-existing gaps (G1–G7) without
  adding any runtime dependency or breaking existing YAMLs / env vars /
  CLI flags / tests.
  - **`armdroid.config.paths`** (G7) — single-source asset-path
    helper. `resolve_asset_path(rel, *, base=None)` walks a 3-stage
    resolution chain (explicit base → `$ARMDROID_REPO_ROOT` →
    pyproject.toml marker walk) plus three SO-101 typed helpers
    (`resolve_so101_urdf`, `…_mjcf_scene`, `…_usd`).
  - **`arm_driver_kind` discriminator** (G5) — new
    `ArmSettings.arm_driver_kind: Literal["mock","esp32"] | None`
    field (PR-B widens to include `"isaac_sim"`). Legacy
    `mock_hardware: bool` is preserved with a once-per-process
    `DeprecationWarning` (suppressible via
    `ARMDROID_SUPPRESS_DEPRECATION=1`); removed in v0.4.0.
    `armdroid.orchestration._driver_kind.resolve_driver_kind()` is
    the precedence helper.
  - **`ArmRLAgentProtocol`** (G4) — new `@runtime_checkable` Protocol
    in `armdroid.domain.protocols` defining `build / train / predict /
    save / load / is_trained / is_built`. The `armdroid.rl_agents`
    registry generic tightens from `type[Any]` to
    `type[ArmRLAgentProtocol]`. `SACAgent` retrofits the protocol
    structurally without source changes; `ArmController` widened to
    accept any conforming agent.
  - **Algorithm-aware `build_arm_controller`** — dispatches via the
    `armdroid.rl_agents` registry instead of importing `SACAgent`
    directly. PR-B can register `rsl_rl_ppo` as a sibling without
    touching factory code. New `sac_her` registry alias matches the
    `ArmTrainingConfig.algorithm` default.
  - **Env + agent telemetry spans** (G6) — 9 new SPAN_* constants
    under `armdroid.telemetry`: `SPAN_ENV_{RESET,STEP,RENDER,CLOSE}`
    and `SPAN_AGENT_{BUILD,TRAIN,PREDICT,SAVE,LOAD}`. Wraps every
    `ArmEnvironmentBase` method (Tower of Hanoi + Laundry Sorting
    inherit) and every `SACAgent` method. `NullTelemetry` default
    keeps this zero-cost via `contextlib.nullcontext`.
  - **Vendored SO-ARM101 simulation assets** (G1, R6) — under
    `assets/so_arm/so101/`: 2 URDFs, 13 STL meshes, 4 MJCF files,
    `LICENSE`, `ATTRIBUTION.md` with pinned upstream commit
    (`fda892cba81032c46c40976a48c9ceadbf40a9ca`,
    Apache-2.0). Repo-root `THIRD_PARTY_NOTICES.md` indexes vendored
    content; `arm.urdf_path` default repaired (was an unresolved path).
    `@field_validator` emits `arm_urdf_path_missing` INFO log when the
    file is absent without raising.
  - **Env-protocol property tests** (G2) — Hypothesis-based parametric
    tests in `tests/property/test_arm_env_invariants.py` over every
    registered env. Verify `reset()` returns `(dict, dict)` with
    canonical goal-conditioned keys; `step()` returns the documented
    5-tuple with reward as Python `float` (not `np.float64`) and
    `terminated/truncated` as Python `bool`.
  - **Conftest migration** — `tests/conftest.py` autouse migrated from
    `ARMDROID_MOCK_HARDWARE=true` to `ARMDROID_ARM_DRIVER_KIND=mock`
    so the new code path is exercised in CI. `delenv` cleanup
    neutralises developer-shell leakage. Tests that explicitly assert
    legacy-bool semantics now use `pytest.warns(DeprecationWarning)`
    to keep the deprecation surface protected until v0.4.0.
  - **Shared `RecordingTelemetry` helper** —
    `tests/helpers/recording_telemetry.py` (promoted from inline class)
    used by every suite asserting on span emission. Establishes
    `structlog.testing.capture_logs` as the documented log-assertion
    pattern (closes peer-review N6).
  - **Pyproject placeholders for PR-B** — new `isaac` pytest marker;
    mypy override block + coverage omit for
    `armdroid.{environments/isaac,hardware/isaac_sim,control.rsl_rl_agent}`;
    tightened ruff per-file ignore list (`["D","ANN","C901","N","RUF012"]`)
    for vendored Isaac task code; `make test/test-all/coverage` exclude
    `isaac` so the default CI matrix never tries to load isaaclab.
  - **R4 / R5 / R7 / R8 (doc-only fixes)** — applied on this branch
    against `docs/research/isaac-omniverse-integration.md`, but the doc
    itself was subsequently reverted on `main` via PR #9 before this
    branch landed. The corresponding implementation-level closures
    (plural `set_joint_position_targets`, gripper rescale-and-invert
    in `gripper.py`, Isaac 5.1 ArticulationCfg) live in PR-B and are
    enumerated in `NEXT_STEPS.md`.

- **Telemetry seam — Axis 6 (Phase 0.3)** — `armdroid.telemetry` package
  with `TelemetryProvider` Protocol, zero-overhead `NullTelemetry` default
  (`contextlib.nullcontext`), and `OtelTelemetry` backend available under
  the new `armdroid[telemetry]` extra (`opentelemetry-api/sdk>=1.27`).
  `SPAN_DRIVER_CONNECT`, `SPAN_DRIVER_DISCONNECT`, `SPAN_DRIVER_SEND`
  constants added; all three are now active in `Esp32JsonDriver`.
- **HIL test hardening (Phase 0.3)** — three new hardware-in-the-loop test
  modules (`test_real_esp32_protocol_edges`, `test_real_esp32_recovery`) and
  a parametrised `test_per_joint_sweep` added to `test_real_esp32_motion`.
  A shared `hil_driver` async fixture is provided in
  `tests/hardware/conftest.py`.  Amplitude configurable via
  `ARMDROID_HIL_SWEEP_AMPLITUDE_RAD`; fault injection gated by
  `ARMDROID_HIL_FAULT_INJECT=1`.
- **HIL runner scripts** — `scripts/run_hil.sh` (bash) and
  `scripts/run_hil.ps1` (PowerShell) for one-command hardware gate with
  optional port override (`ARMDROID_HIL_PORT`).

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
