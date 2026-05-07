# armdroid

Robot arm manipulation platform: SO-ARM100 hardware driver, MuJoCo simulation, SAC+HER policy training, PDDL symbolic planning, and YOLO-based perception for Tower of Hanoi and laundry sorting tasks.

Spun out of [MouseDroidAGI](https://github.com/iancruickshank/Gronk-Droid-Jetson-Nano) (the rover platform that runs on Jetson Orin Nano). armdroid runs on a desktop with a CUDA GPU.

## Hardware

- **Compute**: x86_64 desktop with CUDA GPU (developed against RTX 5060 Ti)
- **Arm**: SO-ARM100 6-DoF over USB serial
- **Camera (optional)**: Intel RealSense D435i for depth perception

## Setup

```bash
git clone https://github.com/iancruickshank/armdroid.git
cd armdroid
python -m venv .venv
source .venv/bin/activate           # Linux / macOS
.venv\Scripts\activate              # Windows

# CUDA-enabled torch (Windows / Linux desktop)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install armdroid + dev tools
pip install -e ".[dev]"

# Optional extras
pip install -e ".[hardware]"        # SO-ARM100 USB driver
pip install -e ".[realsense]"       # depth camera
pip install -e ".[anthropic]"       # LLM replanner backend
```

### Windows: MuJoCo redistributable

`mujoco>=3.0` requires the Microsoft Visual C++ 2015–2022 redistributable. If `import mujoco` fails on a fresh install, download from <https://aka.ms/vs/17/release/vc_redist.x64.exe>.

## Run

`--config` overlays go BEFORE the subcommand. Repeat `--config` to layer multiple overlays.

```bash
# Sim rollout (Tower of Hanoi, mock hardware)
python -m armdroid --config config/tower_of_hanoi.yaml sim --episodes 5

# Train SAC+HER policy
python -m armdroid --config config/training.yaml train

# Rollout against the real arm (requires SO-ARM100 connected)
python -m armdroid --config config/tower_of_hanoi.yaml rollout
```

## Test

```bash
pytest -q tests/
ruff check src/ tests/
mypy --strict src/armdroid/
```

## License

MIT
