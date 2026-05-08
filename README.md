# armdroid

**Robot arm manipulation platform — MakerWorld 6-DoF arm + ESP32, MuJoCo simulation, SAC+HER RL, and PDDL planning.**

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-98.8%25-brightgreen)]()
[![mypy](https://img.shields.io/badge/mypy-strict-blue)]()
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)]()

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

The platform currently targets Tower of Hanoi as the primary training curriculum and is being
extended toward laundry sorting as a transfer task.

---

## Architecture overview

Full C4 diagrams (context, container, and RL controller component) live in
[docs/architecture/C4.md](docs/architecture/C4.md).

### Four-layer stack

| Layer | Package | Responsibility |
|-------|---------|----------------|
| Layer 0 — Perception | `src/armdroid/perception/` | RealSense D435i depth processing, YOLO11 object detection, PnP 6-DoF pose estimation, vision-to-PDDL symbolic state extraction |
| Layer 1 — Symbolic Planning | `src/armdroid/planning/` | PDDL domain + problem generation, Pyperplan BFS/A* search, adaptive replanner with optional Claude API LLM backend |
| Layer 2 — RL Control | `src/armdroid/control/` | SAC+HER goal-conditioned policy (Stable-Baselines3), grasp/place/transit action primitives, trajectory interpolation |
| Layer 3 — Hardware | `src/armdroid/hardware/` | Async MakerWorld 6-DoF arm ESP32 JSON-over-UART driver ([`firmware/arm_esp32/`](firmware/arm_esp32/README.md)), in-memory mock driver for simulation and testing |

All inter-layer boundaries are `@runtime_checkable Protocol` interfaces. `factory.py` is the
single dependency-injection wiring point.

---

## Hardware requirements

| Component | Requirement |
|-----------|-------------|
| CPU / OS | x86_64, Windows 10/11 (Linux also supported) |
| GPU | CUDA-capable GPU; RTX 5060 Ti tested and tuned |
| Robot arm | MakerWorld 6-DoF arm, ESP32 JSON-over-UART (see [firmware/arm_esp32/](firmware/arm_esp32/README.md)) |
| Depth camera | Intel RealSense D435i (optional; mock available) |
| RAM | 16 GB minimum; 32 GB recommended for large replay buffers |
| Storage | ~2 GB for MuJoCo assets, YOLO weights, and checkpoints |

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
https://aka.ms/vs/17/release/vc_redist.x64.exe

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

| File | Purpose |
|------|---------|
| `config/tower_of_hanoi.yaml` | Full Tower of Hanoi preset with mock hardware, 3-disk curriculum |
| `config/training.yaml` | Longer training run tuned for RTX 5060 Ti (5M timesteps, batch 512) |

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

### Run tests

```bash
# Full test suite with coverage
pytest tests/ -v --cov=armdroid --cov-report=term-missing

# Unit tests only (fast)
pytest tests/unit/ -v

# Skip slow tests
pytest tests/ -v -m "not slow"

# Hardware-in-the-loop tests (requires physical arm)
pytest tests/ -v -m hardware
```

Coverage is gated at 85% minimum (`fail_under = 85` in `pyproject.toml`). Current coverage is
98.8%.

### Lint and format

```bash
# Check linting
ruff check src/ tests/

# Auto-fix lint issues
ruff check --fix src/ tests/

# Format check
ruff format --check src/ tests/

# Apply formatting
ruff format src/ tests/
```

### Type checking

```bash
mypy --strict src/armdroid/
```

`mypy --strict` must pass with zero errors. Three modules (`config.schema`, `control.sac_agent`,
`perception.object_detector`) carry targeted `disallow_subclassing_any = false` overrides for
third-party Pydantic / SB3 / Ultralytics base classes.

### CI matrix

The GitHub Actions pipeline runs five stages:

| Stage | Runs on | Python |
|-------|---------|--------|
| Lint (`ruff check` + `ruff format --check`) | ubuntu-latest, windows-latest | 3.11, 3.12 |
| Type check (`mypy --strict`) | ubuntu-latest | 3.11 |
| Test + Coverage (`pytest --cov --cov-fail-under=85`) | ubuntu-latest, windows-latest | 3.11, 3.12 |
| Security (`pip-audit --strict`) | ubuntu-latest | 3.11 |
| Docker (validate `Dockerfile` + `docker-compose`) | ubuntu-latest | — |

---

## Project structure

```
src/armdroid/
  __init__.py              # Version 0.1.0
  __main__.py              # python -m armdroid entry point
  main.py                  # CLI: train | rollout | sim subcommands (argparse)
  protocols.py             # @runtime_checkable Protocol interfaces + data classes
  factory.py               # Single DI wiring point — all build_arm_*() functions
  orchestrator.py          # ArmOrchestrator: train / rollout / shutdown lifecycle

  config/
    schema.py              # ArmSettings(BaseSettings) — single source of truth
    loader.py              # YAML overlay loader (merges multiple files in order)
    logging.py             # LoggingConfig sub-model

  logging/
    setup.py               # structlog configure_logging() + get_logger()

  hardware/
    mock_arm_driver.py     # In-memory mock — no USB required, used in all unit tests
    esp32_json_driver.py   # MakerWorld 6-DoF arm driver (JSON-over-UART / ESP32 + pyserial)

  perception/
    depth_processor.py     # RealSense D435i depth frame processing + hole-filling
    object_detector.py     # YOLO11 on CUDA — detection + NMS
    pose_estimator.py      # PnP 6-DoF pose estimation from 2D detections + depth
    state_extractor.py     # DetectedObject list -> SymbolicState (PDDL predicates)
    facade.py              # ArmPerception — unified ArmPerceptionProtocol facade

  planning/
    pddl_domain.py         # Tower of Hanoi PDDL domain + problem file generation
    symbolic_planner.py    # Pyperplan BFS/A* search backend
    replanner.py           # Adaptive replanner with retry + LLM fallback
    laundry_rules.py       # Laundry sorting PDDL action rules
    llm_replanners/
      anthropic_backend.py # Claude-powered LLM replanner (claude-sonnet-4-6)
      null_backend.py      # No-op fallback replanner

  control/
    sac_agent.py           # SAC + HerReplayBuffer wrapper (Stable-Baselines3)
    controller.py          # ArmController: build_for_env / train_policy / execute_*
    primitives.py          # Grasp / place / transit / home action primitives
    trajectory.py          # TrajectoryGenerator: interpolate, smooth, enforce_limits

  environments/
    base.py                # BaseArmEnv (Gymnasium GoalEnv) with common reward logic
    domain_randomizer.py   # MuJoCo domain randomization (mass, friction, lighting)
    reward_shaping.py      # Configurable reward shaper (grasp/place/collision weights)
    curriculum.py          # Progressive difficulty — stage promotion by success rate
    tower_of_hanoi.py      # TowerOfHanoiEnv — 1–10 disk MuJoCo environment
    laundry_sorting.py     # LaundrySortingEnv — garment classification + sorting

config/
  tower_of_hanoi.yaml      # Full Tower of Hanoi preset (mock hardware, 3-disk)
  training.yaml            # RTX 5060 Ti tuned training run (5M timesteps)

weights/
  arm/                     # SAC+HER policy checkpoints (git-ignored)

tests/
  unit/                    # Per-module unit tests (98.8% coverage)
  integration/             # Cross-layer integration tests
  property/                # Hypothesis property-based tests
```

---

## Roadmap / Next steps

1. **Real hardware integration** — test `Esp32JsonDriver` on the physical arm, calibrate
   joint limits and home position, validate emergency stop.
   See [`firmware/arm_esp32/README.md`](firmware/arm_esp32/README.md) for flashing instructions
   and [`firmware/arm_esp32/PROTOCOL.md`](firmware/arm_esp32/PROTOCOL.md) for the wire protocol.

2. **RealSense D435i integration** — connect `depth_processor.py` against a live camera stream,
   tune `default_focal_length`, `depth_min_m`, and `depth_max_m` from real camera intrinsics.

3. **YOLO training** — collect a Tower of Hanoi disk dataset (varied lighting, backgrounds),
   fine-tune YOLO11, update `arm_perception.yolo_model_path` in config.

4. **MuJoCo scene assets** — author proper MJCF scene XML files (`sim/tower_of_hanoi.xml` and
   `sim/laundry_sorting.xml`) with correct MakerWorld 6-DoF arm geometry and collision meshes.

5. **Training runs** — run SAC+HER curriculum on sim (1-disk → 2-disk → 3-disk → 5-disk),
   instrument with Weights & Biases, track success rate and episode length.

6. **Sim-to-real transfer** — tune domain randomization ranges (`mass_range_pct`,
   `friction_range`, `position_noise_m`, `camera_pose_noise_deg`) by comparing sim and real
   rollouts.

7. **LLM replanner** — validate the Anthropic Claude backend
   (`planning/llm_replanners/anthropic_backend.py`) with real execution-failure scenarios;
   measure replanning latency and success rate.

8. **Integration tests** — add hardware-in-the-loop tests behind `pytest.mark.hardware` that run
   against the real arm with a simple pick-and-place verification.

9. **CI GPU job** — add a GitHub Actions matrix job with `pytest.mark.gpu` on a self-hosted CUDA
   runner; integrate W&B sweep for hyperparameter search (learning rate, HER ratio).

10. **Laundry sorting** — extend the curriculum from Tower of Hanoi to the laundry sorting task:
    train garment detection classes, validate `LaundrySortingEnv`, and define the transfer
    curriculum stages.

---

## License

MIT — see `LICENSE`.

This project is a subsystem of [MouseDroidAGI](https://github.com/ianshank/MouseDroidAGI),
an autonomous Star Wars MSE-6 droid platform running on NVIDIA Jetson Orin Nano.
