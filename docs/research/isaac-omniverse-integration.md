# Isaac Sim / Omniverse Integration for the SO-ARM100

> Status: Research / proposal. No code shipped under this branch beyond this
> document. The mappings to existing armdroid registries are concrete; the
> Isaac Lab task package is documented as a Phase-2b add-on that lands as a
> separate entry-point plugin per [ADR-0003](../architecture/ADR/ADR-0003-registry-pattern.md).

## 1. Scope and goal

armdroid's runtime stack today (`v0.2.x`) trains policies against a MuJoCo
scene and executes them on an ESP32-controlled SO-ARM100-class arm via the
existing `Esp32JsonDriver` ([`firmware/arm_esp32/PROTOCOL.md`](../../firmware/arm_esp32/PROTOCOL.md)).
The configuration even ships referencing `urdf/so_arm100.urdf`
([`config/tower_of_hanoi.yaml:9`](../../config/tower_of_hanoi.yaml)), so the
arm we already target *is* TheRobotStudio
[SO-ARM100 / SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) (5-DoF
arm + 1-DoF gripper, STS3215 servos on a Waveshare bus board).

This document captures how to extend armdroid with an NVIDIA Isaac / Omniverse
backend so that:

1. We can train PPO / SAC policies for the SO-ARM in **Isaac Lab** with PhysX-GPU
   acceleration (10³–10⁴ parallel envs on a single RTX-class GPU) instead of, or
   alongside, the current single-process MuJoCo loop.
2. We get a richer perception stack — RTX path-traced cameras, ground-truth
   segmentation, depth, normals — that we can synthesise into RealSense-D435i
   look-alike data for the existing `armdroid.perception` pipeline.
3. We close the sim-to-real loop end-to-end: Isaac Sim → trained policy
   checkpoint → `ArmController` → `Esp32JsonDriver` → real STS3215 servos,
   with the existing safety / e-stop layer unchanged.

We pin to **Isaac Sim 5.1.0**, **Isaac Lab 2.3.0**, **Python 3.11**. These are
the versions both NVIDIA's documentation and the
[MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101)
reference project converge on as of late 2025 / early 2026, and they match
armdroid's own `requires-python = ">=3.11"`.

## 2. NVIDIA stack — what we actually use

Three layers from NVIDIA's stack are relevant. Anything else (Replicator,
Isaac ROS, GR00T) is optional and out of scope for the first integration.

| Layer | Role | Touchpoint in armdroid |
|---|---|---|
| **Omniverse Kit / OpenUSD** | Scene representation. The arm is stored as a `.usd` articulation; meshes, materials, and physics live as USD prims. | Asset format. Scenes live under `assets/usd/`. |
| **Isaac Sim 5.1.0** | PhysX 5 + RTX renderer wrapped in a Kit app. Provides `omni.isaac.core` Articulation APIs. | Underlying simulator for the new env / driver. |
| **Isaac Lab 2.3.0** | Higher-level RL framework on top of Isaac Sim. Provides `ArticulationCfg`, `ImplicitActuatorCfg`, manager-based `RLEnv`, `gym.register(...)`. | New `armdroid.environments` plugin (Section 6). |

[Isaac Lab](https://isaac-sim.github.io/IsaacLab/) is the only piece that
ships RL primitives. RSL-RL, skrl, and stable-baselines3 are all supported via
config wrappers — armdroid keeps its SAC+HER agent through the existing
`armdroid.rl_agents` registry; we don't have to switch to PPO unless we want
the in-the-box Isaac Lab PPO config.

## 3. SO-ARM100/101 assets — what TheRobotStudio ships

Verified from a clone of [`TheRobotStudio/SO-ARM100`](https://github.com/TheRobotStudio/SO-ARM100):

```
SO-ARM100/
├── Simulation/
│   ├── SO100/           # legacy 5-DoF assets
│   └── SO101/
│       ├── so101_new_calib.urdf      # joints zeroed at midpoint of range (default)
│       ├── so101_old_calib.urdf      # joints zeroed at horizontal extension
│       ├── so101_new_calib.xml       # MuJoCo MJCF for the same model
│       ├── so101_old_calib.xml
│       ├── scene.xml                 # MuJoCo top-level scene (includes one of the calibs)
│       ├── joints_properties.xml
│       └── assets/                   # STL meshes referenced by URDF + MJCF
└── STEP/                             # CAD source
```

The URDF declares **6 movable joints** in this order (verified):

| # | Joint name      | Type     | Limits (rad)             | Notes |
|---|-----------------|----------|--------------------------|-------|
| 1 | `shoulder_pan`  | revolute | `[-1.91986, 1.91986]`    | base yaw |
| 2 | `shoulder_lift` | revolute | `[-1.74533, 1.74533]`    | shoulder pitch |
| 3 | `elbow_flex`    | revolute | `[-1.69, 1.69]`          | elbow pitch (5° calibration offset) |
| 4 | `wrist_flex`    | revolute | `[-1.65806, 1.65806]`    | wrist pitch |
| 5 | `wrist_roll`    | revolute | `[-2.74385, 2.84121]`    | wrist roll (asymmetric range) |
| 6 | `gripper`       | revolute | `[-0.174533, 1.74533]`   | jaw — URDF radians: `0`≈closed, `1.74…`≈open |

> **Gripper convention conflict (R7).** The URDF maps `0` rad → closed and
> `1.74…` rad → open. **armdroid normalises the gripper joint to `[0, 1]`
> with `0` = open and `1` = closed** (verified `domain/state.py:21-22`,
> `firmware/arm_esp32/PROTOCOL.md:40-41`). The Isaac driver must therefore
> **both rescale AND invert** when crossing the URDF-radians ↔
> armdroid-normalised boundary. The single source of truth for this
> rescale-and-invert lives in
> `src/armdroid/hardware/isaac_sim/gripper.py` (PR-B), parametrised on
> `ArmSimIsaacConfig.{gripper_joint_radians_open,
> gripper_joint_radians_closed, gripper_normalised_open,
> gripper_normalised_closed}` so retuning is config-only.

`effort=10`, `velocity=10` on every joint. This matches armdroid's
`arm.dof: 6` and `home_position: [0, 0, 0, 0, 0, 0]`
in `config/tower_of_hanoi.yaml`.

> **Action item for armdroid:** vendor or git-submodule the
> `Simulation/SO101/` directory under `assets/so_arm/so101/` so we have a
> stable, locally pinned copy of the URDF + meshes. The current
> `urdf/so_arm100.urdf` config path is currently *unresolved* — there's no
> file at that path in the repo. Section 5 below lays out the asset layout
> we should adopt.

## 4. Reference Isaac Lab project to build on

[`MuammerBay/isaac_so_arm101`](https://github.com/MuammerBay/isaac_so_arm101)
is a community-maintained Isaac Lab external project that already has working
articulation configs and reach / lift tasks for both SO-ARM100 and SO-ARM101.
It's BSD-3-Clause and is the foundation for the LycheeAI Hub
[*Project: SO-ARM101 × Isaac Sim × Isaac Lab*](https://lycheeai-hub.com/project-so-arm101-x-isaac-sim-x-isaac-lab-tutorial-series)
tutorial series and an
[April 2025 NVIDIA Omniverse Livestream](https://www.youtube.com/watch?v=_HMk7I-vSBQ)
on URDF → OpenUSD with this exact arm. Recommended starting point.

What it gives us out of the box:

- **`ArticulationCfg`** with proven gains:
  - `shoulder_pan` / `shoulder_lift` / `elbow_flex` / `wrist_flex` / `wrist_roll`
    grouped under one `ImplicitActuatorCfg` named `arm`
    (`stiffness` 200/170/120/80/50, `damping` 80/65/45/30/20,
    `effort_limit_sim=1.9`, `velocity_limit_sim=1.5`).
  - `gripper` separate (`stiffness=60`, `damping=20`,
    `effort_limit_sim=2.5`).
  - `init_state.rot=(0.7071, 0, 0, 0.7071)` to lay the arm down on the table.
  - `replace_cylinders_with_capsules=True` for stable contact.
- **Gym tasks already registered** (verified from the source):
  - `Isaac-SO-ARM100-Reach-v0` (training)
  - `Isaac-SO-ARM100-Reach-Play-v0` (eval / smaller scene)
  - `Isaac-SO-ARM101-Reach-v0`
  - `Isaac-SO-ARM101-Reach-Play-v0`
  - `Isaac-SO-ARM100-Lift-v0` and friends (under `tasks/lift/`).
- **RSL-RL PPO config** (`ReachPPORunnerCfg`: 1000 iters, 24 steps/env,
  64×64 actor & critic, ELU, adaptive LR, γ=0.99, λ=0.95, clip 0.2).
- **Driver scripts**: `train`, `play`, `zero_agent`, `random_agent`,
  `list_envs`, all wired to `uv run …`.

We can either:

* (A) take a hard dependency on the package as-is, or
* (B) vendor the `robots/trs_so100/` and `tasks/reach/` modules into a new
  `armdroid.environments.isaac` subpackage with attribution and our own
  joint-name and reward overrides.

Option (B) keeps the licence headers and lets us fold the Isaac task into
armdroid's manager-based env without forcing every developer to install Isaac
Sim. Recommended.

## 5. Proposed asset layout for armdroid

```
assets/
└── so_arm/
    └── so101/
        ├── urdf/
        │   ├── so101_new_calib.urdf      # vendored from TheRobotStudio
        │   └── assets/                   # STL meshes (vendored)
        ├── mjcf/
        │   ├── scene.xml
        │   └── so101_new_calib.xml
        └── usd/
            └── so101.usd                 # produced by Isaac Lab URDF importer
```

All three formats live under one tree so the MuJoCo path keeps working for
fast CI and the Isaac path uses the same physical model.

`config/tower_of_hanoi.yaml` should be updated:

```yaml
arm:
  urdf_path: "assets/so_arm/so101/urdf/so101_new_calib.urdf"
  dof: 6
arm_sim:
  scene_path: "assets/so_arm/so101/mjcf/scene.xml"   # MuJoCo
arm_sim_isaac:
  usd_path:   "assets/so_arm/so101/usd/so101.usd"     # Isaac (new section)
```

## 6. Wiring Isaac Lab into armdroid

armdroid's [Phase-2 plugin seam](../architecture/PHASES.md) was designed for
exactly this: a new env or driver lands as an entry-point registration, not a
core change.

### 6.1 New environment plugin

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
isaac = [
    "isaaclab[all,isaacsim]==2.3.0",
    "torch==2.7.0",
    "torchvision==0.22.0",
]

[project.entry-points."armdroid.environments"]
so_arm_reach_isaac = "armdroid.environments.isaac.reach:SoArmReachIsaacEnv"
so_arm_hanoi_isaac = "armdroid.environments.isaac.hanoi:SoArmHanoiIsaacEnv"
```

`armdroid.environments.isaac.reach` is a thin adapter: it imports the
`SoArm100ReachEnvCfg` from the vendored task and wraps it so that
`isaaclab.envs.ManagerBasedRLEnv(cfg)` looks like an `ArmEnvironmentProtocol`
to the rest of armdroid:

```python
# src/armdroid/environments/isaac/reach.py
from __future__ import annotations
from typing import Any
import gymnasium as gym
from armdroid.domain.protocols import ArmEnvironmentProtocol


class SoArmReachIsaacEnv(ArmEnvironmentProtocol):
    """Adapter: Isaac Lab ManagerBasedRLEnv → armdroid env protocol."""

    def __init__(self, task_cfg, training_cfg, dof: int = 6) -> None:
        # Lazy: importing isaaclab requires the Kit app; only import inside
        # __init__ so unit tests on machines without Isaac Sim still pass.
        from isaaclab.app import AppLauncher
        AppLauncher(headless=True).app  # boot Kit
        self._gym_env = gym.make("Isaac-SO-ARM100-Reach-v0", num_envs=task_cfg.num_envs)

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._gym_env.reset(seed=seed)

    def step(self, action):
        return self._gym_env.step(action)

    def render(self): return self._gym_env.render()
    def close(self) -> None: self._gym_env.close()
```

The two important properties of this adapter:

1. **Lazy import of `isaaclab.app.AppLauncher`** — Isaac Sim must be the first
   thing imported because Kit boots a process-wide Omniverse runtime. Putting
   the import inside `__init__` (not at module top level) means
   `python -m armdroid sim` with the MuJoCo backend still works unchanged on
   machines without Isaac Sim.
2. **No change to `ArmEnvironmentProtocol`** — it already returns
   `dict, float, bool, bool, dict`, which is exactly what
   `ManagerBasedRLEnv` returns.

### 6.2 New `isaac_sim` driver (for HIL-on-sim)

Independently of the env, we can add an `armdroid.drivers` plugin that drives
an Isaac Sim articulation directly through `omni.isaac.core.Articulation` and
exposes the existing `ArmDriverProtocol`. This mirrors the planned
`mujoco_sim` driver in [PHASES.md "Axis 1"](../architecture/PHASES.md):

```toml
[project.entry-points."armdroid.drivers"]
isaac_sim = "armdroid.hardware.isaac_sim:IsaacSimDriver"
```

Behaviourally:

* `connect()` boots `AppLauncher` and loads `assets/so_arm/so101/usd/so101.usd`
  into a default stage; `disconnect()` shuts down the Kit app.
* `send_joint_positions(positions, duration_s)` calls
  `articulation.set_joint_position_targets(positions)` (R4 — Isaac 5.1
  uses the **plural** form; the singular `set_joint_position_target`
  does not exist on `omni.isaac.core.articulations.Articulation`) after
  one trapezoidal-profile interpolation step. Same shape as the ESP32
  firmware-side interpolation, so the controller layer can't tell the
  difference. The gripper component of `positions` is converted from
  armdroid normalised `[0,1]` → URDF radians via
  `armdroid.hardware.isaac_sim.gripper.normalised_to_radians` (the only
  place that rescale-and-invert lives — see Section 3).
* `read_state()` reads `articulation.get_joint_positions()` /
  `get_joint_velocities()` and packs them into the existing `ArmState`.
* `emergency_stop()` zeros the joint targets and freezes the articulation.

That gives us a third deployment target — `mock` (in-memory),
`isaac_sim` (USD + PhysX), `esp32` (real arm) — all swappable through
`cfg.mock_hardware` / `cfg.arm.driver.kind`.

### 6.3 Optional: register Isaac Lab as the RL agent driver

Isaac Lab's RSL-RL runner is an obvious alternative to SAC+HER for short-
horizon reach / lift. If we want it, register a new agent under the existing
`armdroid.rl_agents` group:

```toml
[project.entry-points."armdroid.rl_agents"]
rsl_rl_ppo = "armdroid.control.rsl_rl_agent:RslRlAgent"
```

This is strictly opt-in. SAC+HER stays the default for goal-conditioned
multi-step tasks like Tower of Hanoi.

## 7. URDF → USD conversion (the actual asset pipeline)

Two routes. Both work; pick by use-case.

### Route A — One-shot import via Isaac Sim URDF Importer

Used when we want a hand-tuned USD we can re-edit in the GUI.

1. Launch Isaac Sim 5.1.0.
2. **File → Import** → select `assets/so_arm/so101/urdf/so101_new_calib.urdf`.
3. In the URDF Importer panel:
   - **Fix Base Link**: ✅ (the SO-ARM is bolted to a clamp).
   - **Self-Collision**: ✅.
   - **Joint Drive Type**: `Position`.
   - **Joint Drive Strength** / **Damping**: leave 0; we set them in
     `ArticulationCfg` so they're versioned with code.
   - **Convex Decomposition**: ✅ for collision meshes.
   - **Replace cylinders with capsules**: ✅ (matches Isaac Lab default).
4. **Output path**: `assets/so_arm/so101/usd/so101.usd`.
5. Open the resulting USD, verify joints in **Stage → Properties → Articulation**,
   save.

### Route B — Programmatic via `UrdfFileCfg` (recommended for CI)

Isaac Lab can do the conversion at sim-startup and cache the result. This is
how `MuammerBay/isaac_so_arm101` does it — see
`src/isaac_so_arm101/robots/trs_so100/so_arm100.py`. There's no separate
`.usd` file in the repo; the spawn path is the URDF and Isaac Lab calls the
`UrdfConverter` on first run:

```python
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

SO_ARM100_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        replace_cylinders_with_capsules=True,
        asset_path="assets/so_arm/so101/urdf/so101_new_calib.urdf",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        rot=(0.7071, 0.0, 0.0, 0.7071),
        joint_pos={
            "shoulder_pan": 0.0, "shoulder_lift": 1.57, "elbow_flex": -1.57,
            "wrist_flex": 1.0, "wrist_roll": -1.57, "gripper": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_flex", "wrist_.*"],
            effort_limit_sim=1.9, velocity_limit_sim=1.5,
            stiffness={"shoulder_pan": 200.0, "shoulder_lift": 170.0,
                       "elbow_flex": 120.0, "wrist_flex": 80.0, "wrist_roll": 50.0},
            damping={"shoulder_pan": 80.0, "shoulder_lift": 65.0,
                     "elbow_flex": 45.0, "wrist_flex": 30.0, "wrist_roll": 20.0},
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=2.5, velocity_limit_sim=1.5,
            stiffness=60.0, damping=20.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
```

Joint stiffness / damping numbers above were tuned by Le Lay & Bay against
the real STS3215 servos at 7.4 V; reuse them as-is and re-tune only after
sim-to-real measurement.

## 8. Operator workflow once integrated

```bash
# Install Isaac extras (large download — ~9 GB Kit + PhysX runtime)
pip install -e ".[dev,hardware,isaac]"

# Smoke test: spawn the arm under Isaac Sim, send zero actions
python -m armdroid --config config/tower_of_hanoi_isaac.yaml sim --episodes 1

# PPO reach training, 4096 parallel envs, headless
python -m armdroid --config config/tower_of_hanoi_isaac.yaml \
    train --rl-agent rsl_rl_ppo --num-envs 4096 --headless

# Replay a checkpoint in the GUI
python -m armdroid --config config/tower_of_hanoi_isaac.yaml \
    rollout --checkpoint weights/arm/reach_ppo_latest.pt

# Same checkpoint, real arm, mock hardware off
ARMDROID_MOCK_HARDWARE=false ARMDROID_ARM__SERIAL_PORT=/dev/ttyACM0 \
    python -m armdroid --config config/tower_of_hanoi_isaac.yaml \
    rollout --checkpoint weights/arm/reach_ppo_latest.pt
```

The `rollout` command does **not** know whether the policy was trained in
MuJoCo or Isaac Sim — `ArmController.execute_action` calls into
`ActionPrimitives.transit(...)` which calls `driver.send_joint_positions(...)`
which is the same `Esp32JsonDriver` either way. That's the whole point of
keeping `ArmDriverProtocol` stable.

## 9. Sim-to-real considerations specific to SO-ARM100

| Concern | MuJoCo today | Isaac Sim addition |
|---|---|---|
| **Joint-position scale** | radians, matches URDF | radians, same model — no remap |
| **Gripper convention** | armdroid normalises `[0,1]` with `0`=open, `1`=closed; URDF radians has `0`=closed, `1.74…`=open | adapter must **both rescale AND invert** at the boundary — single source of truth in `armdroid.hardware.isaac_sim.gripper` (PR-B), config-driven via `ArmSimIsaacConfig` |
| **Servo PD gains** | not modelled (kinematic) | Isaac Lab uses implicit PD; tune to STS3215 datasheet (1.9 N·m stall, 60 RPM no-load at 7.4 V) |
| **Camera intrinsics** | mock D435i | RTX renderer can publish ground-truth depth at D435i intrinsics — drop into existing `armdroid.perception` mock |
| **Cycle latency** | 50 ms (firmware interp.) | match by setting `decimation=10` at 200 Hz physics |
| **Domain randomisation** | `arm_sim.domain_randomization=true`, mass/friction/lighting | Isaac Lab `EventTermCfg` randomisers replace this — same config knobs, different backend |

A 5° calibration offset is baked into the SO-ARM URDF on `elbow_flex` (see
the URDF comment we extracted in Section 3). Both the MuJoCo path and the
Isaac path see this offset, so it doesn't drift between sims, but the real
arm has a per-unit servo offset table (`Feetech FT_SCServo_Debug_Qt`) that
must be calibrated before sim-to-real transfer succeeds. This is unchanged
from the current MuJoCo workflow — call it out in the rollout runbook.

## 10. Hardware and licence requirements

* **GPU**: RTX 30-series or newer (Ampere+); RTX 5060 Ti tested for armdroid
  today is sufficient. PhysX-GPU needs ≥8 GB VRAM for 1024 parallel envs at
  the SO-ARM scale.
* **OS**: Ubuntu 22.04 / 24.04, or Windows 10/11. The Isaac Lab container
  image is Ubuntu-only.
* **Driver**: NVIDIA driver ≥535.
* **Disk**: ~15 GB for Isaac Sim 5.1.0 Kit + USD assets + cache.
* **Licence**: Isaac Sim is free for individual use; commercial use requires
  an NVIDIA Omniverse Enterprise subscription. SO-ARM100 is Apache-2.0,
  `MuammerBay/isaac_so_arm101` is BSD-3-Clause — both compatible with
  armdroid's MIT.

## 11. Risks and trade-offs

* **Heavy dependency.** `isaac` extra pulls ~9 GB. Keep it strictly optional
  and never import `isaaclab` at module top level. Adapter modules must
  guard imports inside `__init__`.
* **CI cost.** GitHub Actions doesn't have GPU runners. The Isaac path stays
  out of `make test` / `make check` and gates only on a self-hosted runner
  job tagged `gpu` (we already have the marker registered in `pyproject.toml:124`).
* **Process-singleton Kit app.** Only one `AppLauncher` per Python process.
  `pytest -xdist` parallelism breaks if any worker imports `isaaclab`. The
  `isaac` test suite must run with `-n0`.
* **Joint-naming drift.** SO-ARM101 added a `gripper_link` body; SO-ARM100
  has a `gripper` joint reference. The vendored env cfgs already branch on
  this — don't collapse them.
* **Sim-to-real gap.** Implicit PD ≠ STS3215 dynamics. Plan domain
  randomisation over `stiffness ± 30%`, `damping ± 30%`, `effort_limit ± 20%`
  before the first real-arm rollout.

## 12. Recommended next steps (ordered)

1. Vendor `Simulation/SO101/` into `assets/so_arm/so101/` and fix the
   currently-broken `arm.urdf_path` in `config/tower_of_hanoi.yaml`.
2. Add an `isaac` optional-dependency block + entry-point stubs to
   `pyproject.toml` *without* shipping the implementation — keeps the
   surface area visible for review.
3. Land `armdroid.environments.isaac.reach` as a copy of
   `MuammerBay/isaac_so_arm101/tasks/reach/` with attribution, behind the
   `[isaac]` extra and behind a `gpu` pytest marker.
4. Land `armdroid.hardware.isaac_sim:IsaacSimDriver` matching
   `ArmDriverProtocol`. This is the higher-value piece — it lets us drive
   the same controller against MuJoCo, Isaac, or the real ESP32.
5. Add `config/tower_of_hanoi_isaac.yaml` overlay that switches
   `driver: isaac_sim` and `environment: so_arm_reach_isaac`.
6. Train a reach policy in Isaac, verify zero-shot transfer to the
   ESP32 on the real SO-ARM, and write up the calibration ADR.

Each step ships behind a feature flag in `cfg` and an extras install — no
existing user is forced onto Isaac unless they opt in.

## References

- TheRobotStudio SO-ARM100 / SO-ARM101 — <https://github.com/TheRobotStudio/SO-ARM100>
- SO-ARM101 SO101 simulation README — <https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/README.md>
- SO-ARM101 URDF — <https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf>
- Isaac Lab external project for SO-ARM100/101 (Le Lay & Bay) — <https://github.com/MuammerBay/isaac_so_arm101>
- LycheeAI Hub tutorial series — <https://lycheeai-hub.com/project-so-arm101-x-isaac-sim-x-isaac-lab-tutorial-series>
- NVIDIA Omniverse Livestream — Training a Robot from Scratch (URDF → OpenUSD) — <https://www.youtube.com/watch?v=_HMk7I-vSBQ>
- Isaac Sim URDF Importer extension — <https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/ext_isaacsim_asset_importer_urdf.html> (R8 fix — was 4.5.0; pin matches the 5.1 we ship)
- Isaac Sim Tutorial: Import URDF — <https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/import_urdf.html> (R8 fix)
- Isaac Lab — Importing a New Asset — <https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html>
- Isaac Lab — Adding a New Robot — <https://isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/add_new_robot.html>
- Isaac Lab — Writing an Asset Configuration — <https://isaac-sim.github.io/IsaacLab/main/source/how-to/write_articulation_cfg.html>
- Isaac Lab — Actuators — <https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/actuators.html>
- Isaac Lab — Differential IK / Reach controller — <https://isaac-sim.github.io/IsaacLab/main/source/tutorials/05_controllers/run_diff_ik.html>
- Seeed Studio — SO100Arm Kit in Isaac Sim — <https://wiki.seeedstudio.com/lerobot_so100m_isaacsim/>
- armdroid PHASES — [`docs/architecture/PHASES.md`](../architecture/PHASES.md)
- armdroid ADR-0003 (Registry pattern) — [`docs/architecture/ADR/ADR-0003-registry-pattern.md`](../architecture/ADR/ADR-0003-registry-pattern.md)
