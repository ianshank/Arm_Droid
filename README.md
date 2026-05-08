# armdroid

**Robot arm manipulation platform — MakerWorld 6-DoF arm + ESP32, MuJoCo simulation, SAC+HER RL, and PDDL planning.**

![CI](https://img.shields.io/badge/CI-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)
![mypy](https://img.shields.io/badge/mypy-strict-blue)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)

---

## What is armdroid

armdroid is a hierarchical robot arm training and control platform targeting a MakerWorld 6-DoF
arm controlled over a custom JSON-over-UART protocol running on an ESP32 microcontroller
(see [`firmware/arm_esp32/`](firmware/arm_esp32/README.md) and the
[wire protocol specification](firmware/arm_esp32/PROTOCOL.md)). It trains manipulation policies
entirely in MuJoCo simulation using Soft Actor-Critic with Hindsight Experience Replay (SAC+HER),
then transfers to the physical arm over USB serial.

armdroid grew out of the [MouseDroidAGI](https://github.com/ianshank/MouseDroidAGI) autonomous
rover project. The world-model, reward, memory, curiosity, and safety modules are shared across
both platforms; armdroid adds an arm-specific four-layer stack (perception, symbolic planning, RL
control, hardware execution) and a dedicated MuJoCo training environment.

The current `0.2.x` branch adds an enterprise foundation around that runtime stack: a stable
public API (`armdroid` / `armdroid.api`), a pure domain layer, registry-backed extension seams,
and compatibility shims that keep historical imports working while the package layout is
modernised.

The platform currently targets Tower of Hanoi as the primary training curriculum and is being
extended toward laundry sorting as a transfer task.

---

## Architecture overview

Full C4 diagrams live in [docs/architecture/C4.md](docs/architecture/C4.md), and the design
decisions for the v0.2 refactor live under [docs/architecture/ADR/](docs/architecture/ADR/).

### Layered application model

- Stable API + CLI: `armdroid`, `armdroid.api`, and `src/armdroid/main.py` provide the
  SemVer-stable import surface, CLI entry point, and top-level composition flow.
- Orchestration: `src/armdroid/orchestration/` contains the composition root, lifecycle
  coordination, and factory wiring.
- Adapter subsystems: `src/armdroid/hardware/`, `planning/`, `perception/`, `control/`, and
  `environments/` contain the concrete implementations, registries, and runtime behaviour.
- Configuration: `src/armdroid/config/` provides Pydantic settings, YAML overlays, and logging
  configuration.
- Domain: `src/armdroid/domain/` holds the pure protocols, state types, error hierarchy, and
  units with no internal package imports.
- Compatibility: `src/armdroid/compat/` plus the shim modules preserve legacy imports and method
  names through the migration window.

Built-in drivers, planners, perception backends, environments, and RL agents are exposed through
typed registries backed by PEP 621 entry-point groups. That gives the composition root a plugin
seam without reintroducing hard-coded factory branching.

### Runtime pipeline

- Perception: `src/armdroid/perception/` handles RealSense D435i depth processing, YOLO11
  detection, PnP pose estimation, and symbolic extraction.
- Symbolic planning: `src/armdroid/planning/` handles PDDL generation, Pyperplan BFS/A* search,
  adaptive replanning, and the optional Claude backend.
- RL control: `src/armdroid/control/` owns the SAC+HER policy, action primitives, and trajectory
  interpolation.
- Hardware execution: `src/armdroid/hardware/` contains the ESP32 JSON-over-UART driver, mock
  driver, autodetect, framing, and validation layers.

All subsystem boundaries remain `@runtime_checkable Protocol` interfaces. The composition root in
`armdroid.orchestration.factory` resolves named implementations and shares the driver instance
across the controller and orchestrator.

---

## Hardware requirements

- CPU / OS: x86_64, Windows 10/11 (Linux also supported).
- GPU: CUDA-capable GPU; RTX 5060 Ti tested and tuned.
- Robot arm: MakerWorld 6-DoF arm, ESP32 JSON-over-UART (see
  [firmware/arm_esp32/](firmware/arm_esp32/README.md)).
- Depth camera: Intel RealSense D435i (optional; mock available).
- RAM: 16 GB minimum; 32 GB recommended for large replay buffers.
- Storage: ~2 GB for MuJoCo assets, YOLO weights, and checkpoints.

The depth camera is optional. All perception components fall back to a mock implementation when
`arm_perception.depth_camera_type: "mock"` is set in config (the default in
`config/tower_of_hanoi.yaml`).

---

## Setup

### Prerequisites

- Python 3.11 or 3.12
- CUDA toolkit matching your GPU driver (for YOLO11 inference)
- Microsoft Visual C++ Redistributable 2015-2022 (Windows; required by MuJoCo)

### Installation

```bash
# Clone the repository
git clone https://github.com/ianshank/Arm_Droid.git
cd Arm_Droid

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Install PyTorch with CUDA support first (adjust cu128 to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install armdroid with development dependencies
pip install -e ".[dev]"
```

### Optional extras

```bash
# Real arm driver (ESP32 JSON-over-UART via pyserial)
pip install -e ".[hardware]"

# Intel RealSense D435i depth camera (pyrealsense2)
pip install -e ".[realsense]"

# Claude API LLM replanner (anthropic SDK)
pip install -e ".[anthropic]"

# Install everything
pip install -e ".[dev,hardware,realsense,anthropic]"
```

### Windows note — MuJoCo and Visual C++ Redistributable

MuJoCo 3.x requires the Microsoft Visual C++ Redistributable 2015-2022 (x64). Download and
install it from Microsoft if `import mujoco` fails with a DLL load error:
<https://aka.ms/vs/17/release/vc_redist.x64.exe>

---

## Configuration

armdroid uses a layered YAML configuration system built on Pydantic v2 `BaseSettings`.

### How overlays work

All fields have defaults, so `ArmSettings()` produces a complete working configuration with no
YAML required. Pass one or more `--config` flags to layer YAML overlays on top of those defaults.
Overlays are merged in order; later files win. Environment variables override everything.

```bash
# Base defaults only
python -m armdroid sim

# Tower of Hanoi preset
python -m armdroid --config config/tower_of_hanoi.yaml sim

# Tower of Hanoi + long training run tuned for RTX 5060 Ti
python -m armdroid --config config/tower_of_hanoi.yaml --config config/training.yaml train
```

### Environment variables

All config fields can be overridden via environment variables using the `ARMDROID_` prefix and
`__` as the nested delimiter:

```bash
# Use a different serial port
ARMDROID_ARM__SERIAL_PORT=COM5 python -m armdroid rollout

# Force mock hardware without editing YAML
ARMDROID_MOCK_HARDWARE=true python -m armdroid rollout

# Change number of disks for a single run
ARMDROID_ARM_TASK__NUM_DISKS=5 python -m armdroid sim

# Enable LLM replanner (requires anthropic extra and API key)
ARMDROID_ARM_PLANNING__LLM_REPLANNER_ENABLED=true \
ANTHROPIC_API_KEY=sk-ant-... \
python -m armdroid rollout
```

### Key config files

- `config/tower_of_hanoi.yaml`: full Tower of Hanoi preset with mock hardware and a 3-disk
  curriculum.
- `config/training.yaml`: longer training run tuned for RTX 5060 Ti (5M timesteps, batch 512).

---

## Usage

The `--config` flag(s) must come before the subcommand.

### sim — run the environment without training

Steps the Gymnasium environment for `--episodes` episodes using zero actions. Useful for
verifying the MuJoCo scene, reward shaping, and domain randomization without touching the RL
policy.

```bash
# 5 episodes with default config
python -m armdroid sim

# Tower of Hanoi scene, 10 episodes
python -m armdroid --config config/tower_of_hanoi.yaml sim --episodes 10

# Also available as installed script
armdroid --config config/tower_of_hanoi.yaml sim
```

### train — run the SAC+HER training loop

Builds the orchestrator, wires the `SACAgent` to the environment, runs `SAC.learn()` for the
configured number of timesteps, and saves a checkpoint under `weights/arm/`.

```bash
# Default training run
python -m armdroid --config config/tower_of_hanoi.yaml train

# RTX 5060 Ti tuned run (5M timesteps, batch 512)
python -m armdroid --config config/tower_of_hanoi.yaml --config config/training.yaml train

# Override timesteps on the command line
python -m armdroid --config config/tower_of_hanoi.yaml train --total-timesteps 100000
```

### rollout — plan and execute a Tower of Hanoi solve

Generates a PDDL plan for the configured number of disks and executes each step via the
`ArmController` primitive layer against the real (or mock) driver.

```bash
# 3-disk solve against mock driver
python -m armdroid --config config/tower_of_hanoi.yaml rollout

# 5-disk solve, override from CLI
python -m armdroid --config config/tower_of_hanoi.yaml rollout --num-disks 5

# Real arm driver (ESP32 JSON-over-UART; override mock_hardware flag)
ARMDROID_MOCK_HARDWARE=false \
python -m armdroid --config config/tower_of_hanoi.yaml rollout
```

### Force mock hardware

```bash
python -m armdroid --mock-hardware --config config/tower_of_hanoi.yaml rollout
```

---

## Development

### Recommended local workflow

```bash
# Install with the common local-development extras
make install

# Run the same local gate used before PRs
make check

# Run file-hygiene hooks across the tree
pre-commit run --all-files
```

### Run tests

```bash
# Fast local suite
make test

# Full non-hardware suite
make test-all

# Coverage gate
make coverage

# Focused suites
pytest tests/regression -m regression -v
pytest tests/unit tests/property tests/contract -v --cov=armdroid --cov-report=term-missing

# Hardware-in-the-loop tests (requires physical arm)
pytest tests/ -v -m hardware
```

Coverage is gated at 85% minimum (`fail_under = 85` in `pyproject.toml`). The current branch
validates at 98% total coverage.

### Lint, format, and type checking

```bash
make lint
make typecheck

# Raw commands
ruff check src/ tests/
ruff format --check src/ tests/
mypy --strict src/
```

`mypy --strict` must pass with zero errors. Four modules (`config.schema`,
`control.sac_agent`, `perception.object_detector`, `perception.depth_processor`) carry targeted
`disallow_subclassing_any = false` overrides for third-party Pydantic / SB3 / Ultralytics base
classes.

### GitHub Actions

GitHub Actions currently runs two repository workflows:

- `CI`: Ruff lint/format check, `mypy --strict`, the cross-platform fast test matrix, and the
  Ubuntu full non-hardware suite.
- `firmware-ci`: firmware config codegen drift check, PlatformIO native unit tests, and the ESP32
  firmware build.

---

## Project structure

```text
src/armdroid/
  __init__.py              # Root package re-exporting the stable public API
  __main__.py              # python -m armdroid entry point
  main.py                  # CLI: train | rollout | sim subcommands (argparse)
  _registry.py             # Generic typed registry primitive

  api/                     # Stable SemVer public API
  compat/                  # Deprecation shims + LegacyArmDriverAdapter
  domain/                  # Pure types, errors, protocols, units
  orchestration/           # build_* factories + ArmOrchestrator composition root
  config/                  # Settings submodels, YAML loader, logging config
  logging/                 # structlog configure_logging() + get_logger()

  hardware/
    registry.py            # Driver registry + plugin loading
    mock_arm_driver.py     # In-memory mock — no USB required, used in tests and sim
    esp32/                 # Driver, framing, portfinder, validator
    esp32_json_driver.py   # Backwards-compatible shim for legacy imports

  perception/              # Facade, depth, detection, pose, state extraction, registry
  planning/                # PDDL, planner, replanner, LLM backends, registry
  control/                 # Controller, SACAgent, primitives, trajectory, registry
  environments/            # Task envs, curriculum, randomization, reward shaping, registry

docs/architecture/
  C4.md                    # System/container/component diagrams
  ADR/                     # Architecture decision records for the v0.2 refactor

config/
  tower_of_hanoi.yaml      # Full Tower of Hanoi preset (mock hardware, 3-disk)
  training.yaml            # RTX 5060 Ti tuned training run (5M timesteps)

tests/
  unit/                    # Narrow behavior tests
  regression/              # Public surface + compatibility guardrails
  contract/                # Driver contract tests (all ArmDriverProtocol implementors)
  property/                # Hypothesis invariants
  integration/             # Cross-layer integration tests
  hardware/                # Hardware-in-the-loop tests (pytest.mark.hardware)
  e2e/                     # End-to-end pipeline tests (planner → driver → state)
  helpers/                 # Shared test harness (FakeSerial, PingOnlyFakeSerial, …)
```

---

## Roadmap / Next steps

1. **Real arm validation** — run the decomposed ESP32 driver on the physical MakerWorld arm,
  calibrate joint limits and home pose, and validate emergency-stop / reconnect behaviour.

2. **Complete registry dispatch** — extend the same named-factory lookup used for drivers and
  environments to the remaining planner, perception, and controller composition paths where
  external plugins are expected.

3. **Tighten the typing envelope** — retire the remaining targeted mypy relaxations in
  `config.schema`, `control.sac_agent`, `perception.object_detector`, and
  `perception.depth_processor` as third-party boundaries are wrapped more narrowly.

4. **Broaden public-surface regression coverage** — keep expanding regression tests around
  `armdroid.api`, root-package re-exports, and entry-point/plugin discovery so future internal
  refactors cannot silently change the public contract.

5. **RealSense + YOLO production tuning** — connect the live D435i pipeline, collect task data,
  and tune detector / pose-estimation thresholds from real camera intrinsics and scenes.

6. **7-DoF and sim-to-real follow-through** — finish the MuJoCo scene assets, validate domain
  randomisation against real rollouts, and only then consider making 7-DoF the default config.

---

## License

MIT — see `LICENSE`.

This project is a subsystem of [MouseDroidAGI](https://github.com/ianshank/MouseDroidAGI),
an autonomous Star Wars MSE-6 droid platform running on NVIDIA Jetson Orin Nano.
