# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Hardware integration scaffolding** — `ArmState` dataclass, `ArmDriverError`,
  and `ArmCommandRejected` exceptions in `armdroid.protocols`. These provide
  the value-object and exception contract that the upcoming extended
  `ArmDriverProtocol` surface (lifecycle, latched e-stop, interpolated motion)
  will use.
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
