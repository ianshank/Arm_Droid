# Roadmap: full Isaac integration and full e2e robot integration

This roadmap lays out the logical next steps from the **current landed
state** to two end goals:

1. **Full Isaac Sim integration** — a verified, GPU-validated training
   loop that produces a deployable policy, with a viewer, domain
   randomisation parity, and more than the single `reach` task.
2. **Full end-to-end robot integration** — a closed perception ->
   reason -> safety -> act loop running on the real SO-ARM / ESP32
   hardware, including the foundation-model (Gemini Robotics ER) and
   safety chains scaffolded in Phase A.

A third **convergence** track bridges the two: deploying a
sim-trained policy onto the real arm (sim-to-real).

It supplements (does not replace) [`NEXT_STEPS.md`](../../NEXT_STEPS.md),
which is the canonical per-PR landing log. Where this roadmap references
follow-up IDs (`F2`-`F6`, `SAF-1`, `TD-6`, `TD-7`), those are defined in
`NEXT_STEPS.md`.

---

## Where we are today (baseline)

### Isaac integration — landed

- **PR-B runtime** behind the `[isaac]` extra:
  [`IsaacSimDriver`](../../src/armdroid/hardware/isaac_sim/driver.py)
  (`ArmDriverProtocol`, lazy isaaclab imports, AppLauncher singleton
  guard), [`SoArmReachIsaacEnv`](../../src/armdroid/environments/isaac/reach.py)
  (single-env via [`_TensorAdapter`](../../src/armdroid/environments/_tensor_adapter.py)),
  [`RslRlPpoAgent`](../../src/armdroid/control/rsl_rl_agent.py)
  (RSL-RL `OnPolicyRunner`), and the vendored MuammerBay reach task
  under [`environments/isaac/tasks/reach/`](../../src/armdroid/environments/isaac/tasks/reach).
- **F1 vectorised path** ([ADR-0006](../architecture/ADR/ADR-0006-vec-env-protocol.md)):
  `VecArmEnvironmentProtocol`, `SoArmReachIsaacVecEnv`,
  `RslRlPpoAgent.build_vec/train_vec`, `registry_vec`, and factory
  routing on `cfg.arm_sim_isaac.num_envs > 1`.
- **Blackwell (sm_120) handled**: `isaaclab==2.3.2.post1` pulls
  `torch>=2.7` (sm_120 kernels); see [`docs/GPU_BLACKWELL.md`](../GPU_BLACKWELL.md)
  and `scripts/check_isaac_install.py`.
- **GPU CI**: `gpu-ci.yml` on `workflow_dispatch` / `gpu` label,
  `ubuntu-latest`, asserts the import surface + pure-Python isaac-extra
  unit tests only.

### Isaac integration — NOT yet done

- **Never run on a real GPU.** Risk **R-6** in `NEXT_STEPS.md` is still
  open: `set_joint_position_target` (singular) API, reward tensor shape
  `(num_envs,)`, and URDF/STL import are all *documentation-asserted*,
  not hardware-confirmed. The `tests/isaac/` smoke suite only runs under
  `ARMDROID_ISAAC_RUN=1` on CUDA hardware and has not been executed.
- **No completed training run, no checkpoint, no success-rate metric.**
- **No eval/deploy path**: the CLI exposes `train`/`rollout`/`sim` only
  ([`main.py`](../../src/armdroid/main.py)); there is no `eval` that
  loads an `rsl_rl` checkpoint and reproduces the success rate.
- **Headless only** (F3); `BaseEnv.render` is doc-only (TD-6).
- **Domain-randomisation parity** MuJoCo<->Isaac missing (F4).
- **Only the `reach` task** is vendored; no manipulation tasks.

### E2E robot integration — landed

- **Mock-hardware full DI graph** works end-to-end: orchestrator ->
  perception -> pyperplan planner -> SAC controller -> driver. Covered
  by [`tests/e2e/test_full_pipeline.py`](../../tests/e2e/test_full_pipeline.py).
- **Real ESP32 driver**: serial / TCP / BLE transports + HMAC auth,
  with a gated real-hardware suite under
  [`tests/hardware/`](../../tests/hardware).
- **Perception pipeline** composed in
  [`facade.py`](../../src/armdroid/perception/facade.py): depth ->
  detect -> pose -> symbolic state.
- **Gemini Robotics ER Phase A** scaffolding (ADRs
  [0007](../architecture/ADR/ADR-0007-vision-language-agent-protocol.md) /
  [0008](../architecture/ADR/ADR-0008-open-vocabulary-detector.md) /
  [0009](../architecture/ADR/ADR-0009-safety-middleware.md)): VLA /
  planner / safety / interaction protocols, domain value objects,
  configs, telemetry spans, opt-in extras. **Zero behaviour** — seams
  only.

### E2E robot integration — NOT yet done

- **Gemini Phases B-F** (F2): perception detector + scene reasoner (B),
  `GeminiReplanner` + registry/factory wiring (C), Gemini VLA backend
  (D), Live API (E), Asimov safety chain (F).
- **Safety chain** (Phase F + SAF-1): no `SafetyGuard` actually gates
  actions before the driver yet.
- **Pose orientation** is stubbed (TD-7); detector runs a default
  `ObjectDetector` rather than a validated real model.
- **No closed perception->plan->act loop on real hardware** — the
  ESP32 tests exercise motion/protocol, not the full pipeline.
- **No sim-to-real deployment** of a trained policy.

---

## Track A — Full Isaac Sim integration

> Critical path: **A0 gates everything else.** Nothing downstream is
> trustworthy until the runtime is confirmed on a real Blackwell GPU.

### A0 — GPU bring-up and API verification (BLOCKER)

Run `ARMDROID_ISAAC_RUN=1 pytest tests/isaac -n0` on an RTX 5060 /
5060 Ti and close R-6.

- Confirm `import torch; torch.zeros(1).cuda()` runs an sm_120 kernel
  (per `GPU_BLACKWELL.md`, import success is **not** proof).
- Confirm `Articulation.set_joint_position_target` (singular) exists and
  accepts the `(num_envs, dof)` tensor the driver sends.
- Confirm reward tensor shape is `(num_envs,)`, not `(num_envs, 1)`
  (the `_TensorAdapter` defensive reshape is the fallback).
- Confirm the URDF importer accepts the vendored STL meshes without
  manual convex decomposition; commit the generated USD path handling.

**Exit criteria:** all three `tests/isaac/` smoke suites green on
Blackwell; [ADR-0005](../architecture/ADR/ADR-0005-isaac-sim-backend.md)
verification chain updated from "docs-asserted" to "hardware-confirmed"
with the GPU/driver/CUDA matrix recorded.

### A1 — First training run + checkpoint

Train `RslRlPpoAgent.train_vec` on `so_arm_reach_isaac` to convergence.

- Pin a reproducible training config (seed, `num_envs`, iterations) in
  `config/examples/`.
- Capture reward curve + success rate; define a documented success-rate
  threshold for "trained".
- Persist the checkpoint as a release/LFS artifact (not a git blob) plus
  a committed metrics summary.

**Exit criteria:** reproducible config + checkpoint artifact + metrics;
success rate above the agreed threshold.

### A2 — Policy eval + deploy path

Wire checkpoint load -> inference action.

- Add an `armdroid eval` CLI subcommand: load an `rsl_rl` checkpoint,
  run deterministic eval through the **single-env protocol path**
  (`SoArmReachIsaacEnv` / `IsaacSimDriver`).
- Add a documented checkpoint accessor on `RslRlPpoAgent`
  (no private reach-through).

**Exit criteria:** `armdroid eval --checkpoint ...` reproduces the A1
success rate within tolerance on a single env.

### A3 — Viewer + video (F3, TD-6)

- Non-headless GUI launch path (the driver already threads `headless`
  through `ArmSimIsaacConfig`).
- Implement `render()` via `mujoco.Renderer` offscreen framebuffer for
  MuJoCo and the Isaac camera path for episode video.

**Exit criteria:** `armdroid sim --render` produces an episode video;
GUI viewer launches on a workstation.

### A4 — Domain-randomisation parity (F4)

- Shared DR config (mass / friction / lighting ranges) driving both
  the MuJoCo `domain_randomizer` wrapper and the Isaac task cfg.

**Exit criteria:** one DR config block drives both backends; a parity
regression test asserts the same ranges are applied.

### A5 — Task breadth beyond `reach`

- Vendor / author a manipulation task (pick-and-place, then
  peg-in-hole) following the reach registry + `mdp/` pattern. No
  `if task_type ==` ladders — extend `_VEC_TASK_REGISTRY_NAMES`.

**Exit criteria:** at least one manipulation task trains and evals
through the same A1/A2 path.

---

## Track B — Full e2e robot integration

> These can proceed in parallel with Track A (different hardware:
> camera + ESP32 arm, not a training GPU). B0 and B4 are the highest
> value: real perception in, safe actions out.

### B0 — Real perception (close TD-7)

- Integrate a validated detector model (the ADR-0008 open-vocab path)
  in place of the default `ObjectDetector`.
- Implement OpenCV PnP for pose **orientation** (TD-7): per-class 3D
  model points + 2D detections + calibrated intrinsics.

**Exit criteria:** `detect_objects()` returns true 6-DoF poses on real
RGB-D frames; integration test against recorded frames.

### B1 — Gemini Phase B: perception detector + scene reasoner (F2-B)

- Frame buffer + scene reasoner emitting `SceneInsight` from Gemini ER
  1.6, behind the `[gemini]` extra.

**Exit criteria:** `SceneInsight` produced from real frames; falls back
cleanly when the extra is absent.

### B2 — Gemini Phase C: replanner registry + factory (F2-C)

- `GeminiReplanner` registered in the LLM-replanner registry; factory
  wiring; symbolic-planner fallback on replanner failure.

**Exit criteria:** replanner swaps into the planning registry via config
discriminator; rollout still completes when Gemini is unavailable.

### B3 — Gemini Phase D: VLA backend (F2-D)

- Gemini `VisionLanguageAgentProtocol` backend producing `ArmAction`;
  controller `isinstance` dispatch (VLA vs RL) — the disjoint method
  names from ADR-0007 already make this unambiguous.

**Exit criteria:** VLA produces `ArmAction`; controller routes VLA vs RL
agents without a type ladder.

### B4 — Safety chain: Phase F + SAF-1

- `SafetyGuardProtocol` Asimov chain gating **every** `ArmAction`
  before it reaches the driver; `Verdict` logged via telemetry; e-stop
  integration.
- NEISS safety corpus, OEM velocity profiles, operator custom rules
  (SAF-1).

**Exit criteria:** no action reaches the driver without a `Verdict`;
unsafe actions are rejected and logged; e-stop path tested.

### B5 — Gemini Phase E: Live API (F2-E, optional)

- `InteractionSessionProtocol` streaming session behind `[gemini-live]`.

**Exit criteria:** interaction session streams events; cleanly optional.

### B6 — Full closed loop on the real ESP32 arm

- Orchestrator rollout on real hardware: camera -> perception ->
  (scene reasoner / replanner) -> safety -> controller -> ESP32 driver.

**Exit criteria:** a gated e2e hardware test completes a pick task end
to end on the physical arm.

---

## Track C — Convergence: sim-to-real

### C0 — Sim-trained policy on the real arm

Depends on **A2** (deployable checkpoint), **A4** (DR for the reality
gap), and **B4** (safety gating).

- Deploy the A1/A2 Isaac-trained checkpoint onto `IsaacSimDriver` eval,
  then onto the ESP32 arm via the shared `ArmDriverProtocol` + the
  gripper normalisation in
  [`gripper.py`](../../src/armdroid/hardware/isaac_sim/gripper.py).
- Document the reality gap and the DR mitigation.

**Exit criteria:** the sim policy executes `reach` on the real arm
within the B4 safety envelope; reality-gap notes recorded in an ADR.

---

## Cross-cutting / housekeeping

- **CI:** decide on a self-hosted GPU runner (currently explicitly
  avoided). Until then A0/A1 are run manually and the results pinned in
  ADR-0005.
- **v0.4 cleanups:** F5 (tighten `ArmEnvironmentProtocol.step` return
  type), F6 (remove the deprecated `mock_hardware` bool — update every
  `ArmSettings(mock_hardware=...)` call site in the same PR).
- **Attribution:** keep `THIRD_PARTY_NOTICES.md` in step with any new
  vendored task code (R-2).

---

## Dependency-ordered milestone view

```
Isaac:   A0 ──> A1 ──> A2 ──┬──> A3
                            ├──> A4 ─┐
                            └──> A5  │
                                     │
Robot:   B0 ──> B1 ──> B2 ──> B3     │
          └───────────────> B4 ──────┼──> C0  (sim-to-real)
                            B5        │
          B0..B4 ─────────> B6 ──────┘
```

- **A0 is the single hard blocker** for the entire Isaac track.
- **B0 and B4** are the highest-leverage robot milestones and unblock
  both B6 (real closed loop) and C0 (sim-to-real).
- **C0** is the true "full e2e" terminus: it requires a deployable
  policy (A2), reality-gap mitigation (A4), and safety gating (B4).

## Suggested phasing

1. **Phase 1 (unblock):** A0, B0 — both need only hardware access, no
   new architecture.
2. **Phase 2 (capability):** A1 + A2 (train + deploy), B1 + B2 + B3
   (Gemini perception/replan/VLA).
3. **Phase 3 (safety + breadth):** B4 (safety chain), A4 (DR parity),
   A3 (viewer), A5 (tasks).
4. **Phase 4 (converge):** B6 (real closed loop), C0 (sim-to-real),
   B5 (Live API), then the v0.4 cleanups (F5/F6).
