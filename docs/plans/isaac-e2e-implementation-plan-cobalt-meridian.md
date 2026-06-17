# Implementation plan: full Isaac integration + full e2e robot integration

> **Plan ID:** `cobalt-meridian`
> **Source roadmap:** [`docs/roadmap/isaac-and-robot-integration.md`](../roadmap/isaac-and-robot-integration.md)
> **Branch:** `claude/isaac-robot-integration-roadmap-yPhVs`
> **Method:** subagent-driven, sequential, dependency-ordered. Every
> milestone is backward-compatible, registry/config-driven, fully
> tested, logged + traced, and ends with a gap-analysis/cleanup pass.

This is the executable companion to the roadmap. The roadmap says
*what* and *why*; this plan says *which files*, *which tests*, *which
subagent*, and *what "done" means* — grounded in a deep read of the
current code (file:line references throughout).

---

## 0. Operating principles (apply to every task)

1. **Backward compatible.** No protocol break without a sibling
   protocol + deprecation cycle (the ADR-0006 `VecArm*Protocol`
   precedent). Widen `Literal`s additively; keep legacy adapters until
   a named major (v0.4 for `mock_hardware`).
2. **No hardcoded values.** Every numeric/string knob is a Pydantic
   field with `ge/le/gt` bounds. This plan *fixes existing violations*
   (see Task X1) as well as forbidding new ones.
3. **Registry + entry-point, never `if kind ==` ladders.** New backends
   register in the matching `registry.py` + `pyproject.toml`
   entry-point group, mirrored by a regression test
   (`tests/regression/test_entry_point_mirror.py`).
4. **Reuse fakes before `MagicMock`.** Catalog in §Testing. Build the
   *missing* fakes once, share across phases.
5. **Logging + telemetry are not optional.** `get_logger(__name__)`
   structured events on every state transition; `get_telemetry()
   .start_span(SPAN_*, **attrs)` on every external call. Span name
   constants live in `armdroid.telemetry` only.
6. **Every task ends green** on the three gates:
   ```bash
   python -m ruff check src/ tests/ --exclude tests/helpers/fake_serial.py
   python -m mypy --strict src/
   python -m pytest -m "not hardware and not isaac" --cov=armdroid --cov-fail-under=85 -q
   ```
7. **Gap-analysis pass per milestone:** delete dead code, collapse
   duplication, refresh stale comments, confirm no orphan config field
   / unused entry point.

### Subagent / tooling assignment

| Role | Agent / skill | Used for |
| --- | --- | --- |
| Architecture design per milestone | `Plan` agent | turning a task into a file-level design before coding |
| Code search / "where is X" | `Explore` agent | locating call sites, registries, fixtures |
| Implementation | `general-purpose` agent (or direct edits) | writing code + tests on the branch |
| Peer review of each diff | `code-review` skill (`/code-review`) | correctness pass before commit |
| Security pass on HW/LLM/safety code | `security-review` skill | safety chain, transport auth, LLM input handling |
| Manual feature validation | `verify` / `run` skill | CLI subcommands, viewer, real-HW smoke |
| Permission-prompt reduction | `fewer-permission-prompts` | once the loop of allowed commands stabilises |

Dispatch pattern (per non-trivial task): **`Plan` → implement →
`/code-review` → fix → commit**. Independent tasks within a phase are
dispatched to parallel subagents.

---

## Track A — Full Isaac Sim integration

### A0 — GPU bring-up + API verification *(BLOCKER, manual + GPU)*

**Objective.** Close risk R-6; nothing downstream is trustworthy until
the runtime runs on a Blackwell GPU.

**Why it can't be automated here.** `tests/isaac/conftest.py` gates on
`ARMDROID_ISAAC_RUN=1` + a CUDA GPU; `docs/GPU_BLACKWELL.md` documents
that *import success is not proof* (sm_120 kernel must actually run).

**Steps.**
1. On an RTX 5060 / 5060 Ti: `pip install -e ".[dev,isaac]" --extra-index-url https://pypi.nvidia.com`, run `scripts/check_isaac_install.py`, then `ARMDROID_ISAAC_RUN=1 pytest tests/isaac -n0`.
2. Confirm the three unverified assumptions against live code:
   - **Joint API:** `driver.py:362` calls singular `set_joint_position_target`. The header (`driver.py:8-11`) documents this as the Isaac Lab 2.3 runtime form (plural is the legacy `omni.isaac.core` class). *Status today: docs-asserted, not GPU-confirmed.*
   - **Reward shape:** `_tensor_adapter.py` already defends `(num_envs,)` vs `(num_envs,1)` via `reshape(-1)[0]`. Confirm which shape is real and drop the defensive branch only if proven unnecessary (otherwise keep + comment).
   - **URDF/STL import:** `articulation.py` `UrdfFileCfg(asset_path=...)` — confirm meshes import without manual convex decomposition; capture the generated `usd_path` handling (`sim_isaac.py` `usd_path` field).
3. Record the GPU/driver/CUDA matrix.

**Deliverable.** Update [ADR-0005](../architecture/ADR/ADR-0005-isaac-sim-backend.md)
verification section from "docs-asserted" → "hardware-confirmed";
append results to the roadmap baseline.

**Exit criteria.** All three `tests/isaac/` smoke suites green on
Blackwell; ADR-0005 updated.

**Subagent.** `verify` skill drives the smoke run + observation; human
provides the GPU box.

---

### A1 — First training run + checkpoint *(GPU)*

**Objective.** Produce a converged `reach` policy + reproducible config
+ metrics.

**Grounding.** `RslRlPpoAgent.save()` / `.load()` already exist
(`control/rsl_rl_agent.py`), default checkpoint path derives from
`ArmTrainingConfig.weights_dir`. Vec training is `train_vec`; factory
routes `num_envs > 1` via `_VEC_CAPABLE_ALGORITHMS` /
`_VEC_TASK_REGISTRY_NAMES` (`orchestration/factory.py`).

**Files.**
- `config/examples/isaac_reach_train.example.yaml` *(new)* — pinned seed, `num_envs`, iterations; documented + regression-tested like the other example overlays.
- `assets/policies/` *(new, LFS/release — not a git blob; add to `.gitignore` like the USD artefact rule in ADR-0005 decision 9)*.
- `docs/training/reach-baseline.md` *(new)* — reward curve + success-rate metric + exact repro command.

**Exit criteria.** Reproducible config + checkpoint artefact + committed
metrics; success rate above an agreed threshold (recorded in the doc).

---

### A2 — Policy eval + deploy path

**Objective.** `armdroid eval --checkpoint ...` loads an rsl_rl
checkpoint and reproduces the A1 success rate through the **single-env
protocol path**.

**Grounding (gap).** CLI today exposes only `train`/`rollout`/`sim`
(`main.py:57-110`). `RslRlPpoAgent` has `load()` + `predict()` but **no
CLI/orchestrator entry wires load→predict for eval**. Eval should
bypass perception/planner (not needed) but reuse `build_arm_controller`
which already threads `cfg.arm_rsl_rl_ppo` on `algorithm == "rsl_rl_ppo"`.

**Files / changes.**
- `main.py` — add `eval` subparser (`--checkpoint` required, `--episodes` default from config, optional `--num-envs`); add `_eval(cfg, checkpoint, episodes)` dispatch. *(additive; no existing subcommand touched.)*
- `control/rsl_rl_agent.py` — if `predict` needs a built-but-untrained runner, add a documented `load_for_eval(path)` accessor (no private `_runner` reach-through — mirror the ADR-0006 `as_runner_env()` discipline).
- Telemetry: emit a new `SPAN_AGENT_EVAL` constant (add to `armdroid.telemetry`, pin in `tests/unit/telemetry/test_span_constants.py`).

**Testing.**
- `tests/unit/control/test_rsl_rl_eval.py` — `load_for_eval` + `predict` with `make_fake_isaac_env` (no GPU).
- `tests/unit/scripts/` or `tests/unit/test_cli.py` — arg parsing for `eval`.
- `tests/isaac/test_eval_reproduces_success.py` — gated GPU test reproducing A1 rate ±tolerance.

**Exit criteria.** `armdroid eval` reproduces A1 success rate within
tolerance on a single env; unit path green without a GPU.

---

### A3 — Viewer + episode video *(F3, TD-6)*

**Objective.** Non-headless GUI + offscreen render → episode video.

**Grounding.** `headless` is threaded through `ArmSimIsaacConfig` →
`AppLauncher(headless=...)` (`driver.py:215`), `step(render=not headless)`
(`driver.py:261`), env `render()` returns `None` when headless
(`reach.py`). `BaseEnv.render` is doc-only (TD-6; `environments/base.py`).
Isaac offscreen capture needs an EGL/camera framebuffer path.

**Files.**
- `environments/base.py` — implement MuJoCo `render()` via `mujoco.Renderer` offscreen framebuffer.
- `environments/isaac/reach.py` — wire Isaac camera capture when `enable_cameras` (already a config field).
- `main.py` `sim` subcommand — add `--render` flag + video sink path (config-driven output dir; no hardcoded path).
- New config fields on `ArmSimIsaacConfig`: `dome_light_intensity` (replaces hardcoded `2000.0` at `driver.py:240` — see Task X1).

**Testing.** `tests/unit/environments/test_render.py` (MuJoCo offscreen,
CPU-only); Isaac video path gated `@pytest.mark.isaac`.

**Exit criteria.** `armdroid sim --render` produces a video; GUI viewer
launches on a workstation.

---

### A4 — Domain-randomisation parity *(F4)*

**Objective.** One DR config drives both MuJoCo and Isaac.

**Grounding (gap).** MuJoCo DR lives in `environments/domain_randomizer.py`
(`mass_range_pct`, `friction_range`, `position_noise_m`,
`lighting_intensity_range`) consuming `ArmSimConfig`. `ArmSimIsaacConfig`
has **zero DR fields**; the reach task has event-based reset randomisation
(`reach_env_cfg.py` `EventCfg` `reset_joints_by_scale`).

**Design.** Extract a shared `DomainRandomizationConfig` (new
`config/schema/domain_rand.py`) referenced by *both* `ArmSimConfig` and
`ArmSimIsaacConfig` (composition, not duplication). Thread Isaac side
into the task `EventCfg`.

**Testing.** `tests/regression/test_dr_parity.py` — asserts the same
config block yields the same ranges on both backends.

**Exit criteria.** Shared DR config; parity regression green.

---

### A5 — Task breadth beyond `reach`

**Objective.** Add pick-and-place (then peg-in-hole) with no
`if task_type ==` ladder.

**Grounding.** `reach` registers in `environments/registry.py` +
`registry_vec.py` + `pyproject.toml` entry points + factory's
`_VEC_TASK_REGISTRY_NAMES`. Task cfg layout: `ReachSceneCfg` /
`CommandsCfg` / `ActionsCfg` / `ObservationsCfg` / `EventCfg` /
`RewardsCfg` under `tasks/reach/`.

**Files (clone-and-modify pattern).**
- `environments/isaac/tasks/pick_place/{__init__,pick_place_env_cfg}.py` + `mdp/`.
- Register `so_arm_pick_place_isaac` (+ `_vec`) in both registries; add entry points; extend `_VEC_TASK_REGISTRY_NAMES`.
- Widen `ArmTaskConfig.task_type` Literal additively.

**Testing.** `tests/isaac/test_smoke_pick_place.py` (gated); registry
mirror regression updated.

**Exit criteria.** ≥1 manipulation task trains + evals through the
A1/A2 path.

---

## Track B — Full e2e robot integration

> All of B can proceed in parallel with A (camera + ESP32 arm, not a
> training GPU). **B0 and B4 are highest leverage.** Phase A scaffolding
> (protocols, value objects, configs, telemetry spans, extras) already
> exists per ADRs 0007/0008/0009 — these tasks add the *concrete
> backends + wiring* the scaffolding reserved.

### B0 — Real perception, close pose orientation *(TD-7)*

**Objective.** `detect_objects()` returns true 6-DoF poses on real
RGB-D.

**Grounding (gap).** `pose_estimator.py` hardcodes orientation to
`np.zeros(3)` with `orientation_unmeasured=True` in the log; docstring
lists the PnP prerequisites (per-class 3D model points, 2D detections,
calibrated intrinsics). `ObjectDetector` is a *real* ultralytics YOLO
load (not a stub) but falls back to `[]` when the model is absent.

**Files.**
- `perception/pose_estimator.py` — OpenCV `solvePnP`; per-class 3D model points sourced from URDF/mesh via a config-driven map (no hardcoded keypoints).
- `config/schema/perception.py` — add `model_points_path` / per-class keypoint config.

**Testing.**
- New fake: `tests/helpers/fake_camera.py` (RGB-D source) — *missing today*, build once, reuse for B1/B6.
- `tests/unit/perception/test_pose_pnp.py` against recorded frames + known pose.
- `tests/integration/test_perception_pose.py`.

**Exit criteria.** True 6-DoF poses on recorded RGB-D; PnP unit test
green.

---

### B1 — Gemini Phase B: scene reasoner *(F2-B)*

**Objective.** `SceneInsight` from Gemini ER on real frames, behind
`[gemini]`.

**Grounding (gap).** `SceneInsight` value object exists
(`domain/state.py`); `SPAN_PERCEPTION_SCENE_REASON` declared but never
emitted; **no `SceneReasonerProtocol`, no backend, no registry**.

**Files.**
- `domain/protocols.py` — add `SceneReasonerProtocol` (sibling to `HighLevelPlannerProtocol`; disjoint method names per ADR-0007 discipline).
- `perception/scene_registry.py` *(new)* + entry-point group `armdroid.scene_reasoners`.
- `perception/scene_reasoners/gemini_backend.py` *(new)* — lazy `google.genai` import inside the `[gemini]` extra; emits `SPAN_PERCEPTION_SCENE_REASON`.
- Factory wiring: `ArmPerceptionConfig` discriminator selects the reasoner; default = null/no-op (backward compatible).

**Testing.** Reuse `fake_gemini_sdk()` (already in `tests/helpers/fake_llm_sdk.py`); `tests/contract/test_scene_reasoner_contract.py`.

**Exit criteria.** `SceneInsight` produced from frames; clean fallback
when extra absent.

---

### B2 — Gemini Phase C: replanner registry + Gemini backend *(F2-C)*

**Objective.** `GeminiReplanner` swaps into the planning registry;
symbolic fallback on failure.

**Grounding.** `LLMReplannerConfig.backend` Literal **already includes
`"gemini"`** (`config/schema/llm.py`) with no backend behind it.
`AnthropicReplanner` is the working reference. **Duplicate retry logic**
exists in `planning/replanner.py` and `llm_replanners/anthropic_backend.py`
— extract a shared retry/backoff utility (cleanup, see Task X2).

**Files.**
- `planning/llm_replanners/gemini_backend.py` *(new)* — implements `LLMReplannerProtocol` (`name`, `replan`), lazy import guard like Anthropic.
- `planning/llm_replanners/_retry.py` *(new shared util)* — collapse the two duplicate backoff loops.
- Backend dispatch by `cfg.backend` name (registry, not `if`).

**Testing.** `tests/contract/test_replanner_contract.py` parametrised
across `null`/`anthropic`/`gemini` via `fake_*_sdk`.

**Exit criteria.** Replanner selected by config; rollout still completes
when Gemini unavailable (null fallback).

---

### B3 — Gemini Phase D: VLA backend + controller dispatch *(F2-D)*

**Objective.** Gemini VLA produces `ArmAction`; controller routes VLA
vs RL with no type ladder.

**Grounding (gap).** `VisionLanguageAgentProtocol` exists with disjoint
methods (`build_vla`, `act`, `is_vla_built`, `is_vla_loaded`).
`controller.py` accepts **only** `ArmRLAgentProtocol` and has no VLA
branch. `ArmVlaConfig` exists (`enabled=False`, `model=
"gemini-robotics-er-1.6"`, all bounded). No backend, no
`armdroid.vision_agents` registry.

**Files.**
- `control/vision_agents/gemini_vla.py` *(new)* — `VisionLanguageAgentProtocol` backend, lazy `[gemini-vla]` import.
- `control/controller.py` — widen agent slot to `ArmRLAgentProtocol | VisionLanguageAgentProtocol`; `isinstance` dispatch (ADR-0007 makes it unambiguous); add `execute_vla_step`.
- `control/registry.py` + entry-point group `armdroid.vision_agents`.
- `orchestration/factory.py` — `build_arm_controller` VLA branch.

**Testing.**
- New fake: `tests/helpers/fake_vla_agent.py` — *missing today*.
- `tests/regression/test_rl_agent_does_not_match_vla_protocol.py` — ADR-0007 references this but it **does not exist**; add it.
- `tests/contract/test_vla_agent_contract.py`.

**Exit criteria.** VLA emits `ArmAction`; controller dispatch covered by
isinstance regression.

---

### B4 — Safety chain *(Phase F + SAF-1, HIGH leverage)*

**Objective.** No action reaches the driver without a `Verdict`.

**Grounding (gap).** `SafetyGuardProtocol` (`check_plan`, `check_action`)
+ `Verdict` value object exist; **no concrete guard, no chain composer,
no gate wired**. `SPAN_SAFETY_CHECK_PLAN/ACTION/DENY` declared but never
emitted. NEISS asset slot reserved (`pyproject.toml`
`force-include "assets/safety"`); `ArmSafetyConfig.neiss_corpus_path`
defaults set but **corpus not shipped**.

**Design (defence in depth).**
- Plan-level gate in `orchestrator.rollout()` after `plan()` (`orchestrator.py:174`).
- Action-level gate in `controller.execute_*` before primitive/driver dispatch.
- Composable `SafetyChain` (ordered guards; first deny wins; every `Verdict` logged + `SPAN_SAFETY_*` emitted).

**Files.**
- `safety/` package *(new)*: `chain.py`, `guards/velocity.py`, `guards/workspace.py`, `guards/neiss.py`; registry + `armdroid.safety_guards` entry-point group.
- `assets/safety/neiss_corpus.json` *(new asset, versioned per SAF-1 cadence)*.
- Gate calls in `orchestrator.py` + `controller.py` (default chain = permissive no-op guard so existing rollouts are byte-identical until a guard is configured → backward compatible).

**Testing.**
- New fakes: `tests/helpers/fake_safety_guard.py`, fake NEISS corpus.
- `tests/contract/test_safety_guard_contract.py`; `tests/unit/safety/*`; property test that an unsafe action is *always* denied.
- **`security-review` skill** on the whole package.

**Exit criteria.** Every action gated; unsafe actions denied + logged;
e-stop path tested; security review clean.

---

### B5 — Gemini Phase E: Live API *(F2-E, optional)*

**Objective.** `InteractionSessionProtocol` streaming behind
`[gemini-live]`.

**Files.** `interaction/gemini_live.py` *(new)*, emits
`SPAN_INTERACTION_*`. **Cleanly optional** — absent extra = feature off.

**Testing.** `tests/unit/interaction/` with a fake streaming SDK.

**Exit criteria.** Session streams events; optional.

---

### B6 — Full closed loop on real ESP32 arm *(HIL)*

**Objective.** camera → perception → (scene reasoner / replanner) →
safety → controller → `Esp32JsonDriver`, on hardware.

**Grounding.** `Esp32JsonDriver` fully implements the protocol
(connect/disconnect/send/read_state, transports, validator). The gap is
*orchestration wiring*: a frame-capture loop, scene-reasoner call,
safety gate, and (optional) VLA branch — all delivered by B0–B4.

**Files.** `orchestration/` — assemble the real-HW rollout from the
B0–B4 pieces; CLI `rollout --hardware` path (config-driven, gated).

**Testing.** `tests/hardware/test_closed_loop_pick.py`
(`@pytest.mark.hardware`, auto-skip without `ARMDROID_HIL_RUN=1`,
reuse `hil_driver` fixture + `fake_camera` for the perception side if no
real camera).

**Exit criteria.** Gated HIL test completes a pick task end-to-end.

---

## Track C — Convergence: sim-to-real

### C0 — Sim-trained policy on the real arm

**Depends on A2 (deployable checkpoint), A4 (DR for the reality gap),
B4 (safety gating).**

**Design.** Load the A2 checkpoint; run eval on `IsaacSimDriver`, then
deploy onto `Esp32JsonDriver` through the shared `ArmDriverProtocol` +
the existing gripper normalisation (`hardware/isaac_sim/gripper.py`
normalised↔radians). Safety chain (B4) wraps every command.

**Files.** `docs/architecture/ADR/ADR-0010-sim-to-real.md` *(new)* —
reality-gap notes + DR mitigation; `eval --deploy esp32` path.

**Exit criteria.** Sim policy executes `reach` on the real arm within
the B4 safety envelope; reality gap documented.

---

## Cross-cutting tasks

### X1 — Remove existing hardcoded values *(no-hardcoded-values cleanup)*

Gap analysis found three literals in Isaac code not routed through
config — fix as part of A2/A3:
- `articulation.py:115` `max_depenetration_velocity = 5.0` → new `ArmSimIsaacConfig.max_depenetration_velocity`.
- `driver.py:240` dome light `2000.0` → `ArmSimIsaacConfig.dome_light_intensity`.
- `driver.py:398` is-moving threshold `1e-6` → `ArmSimIsaacConfig.velocity_moving_threshold`.

(Internal asyncio scheduling floors in `esp32/driver.py` and
`_TARGET_DIMS=3` in `orchestrator.py` are *not* user-tunable — leave,
they are documented invariants.)

### X2 — Collapse duplication

- Shared retry/backoff util (`planning/llm_replanners/_retry.py`) used
  by `replanner.py` + `anthropic_backend.py` + new `gemini_backend.py`
  (B2).
- `robots.py` `SO_ARM100_CFG` / `SO_ARM101_CFG` are currently identical
  — either parameterise the real SO-101 calibration or document the
  alias explicitly (follow-up flagged in code).

### X3 — Test-infrastructure build-out *(do early — unblocks B)*

New reusable fakes (build before the phases that need them):
- `tests/helpers/fake_camera.py` (RGB-D source) — B0/B6.
- `tests/helpers/fake_vla_agent.py` — B3.
- `tests/helpers/fake_safety_guard.py` + fake NEISS corpus — B4.

Missing tests referenced by ADRs but absent today (add):
- `tests/regression/test_rl_agent_does_not_match_vla_protocol.py` (ADR-0007).
- DetectedObject positional-ctor compat regression (ADR-0008).

### X4 — CI lanes + coverage

- `.github/workflows/gemini-ci.yml` *(new, analog of `gpu-ci.yml`)* —
  path-triggered on `planning/llm_replanners/**`, `perception/**`,
  `control/vision_agents/**`, `safety/**`; installs `[dev,gemini]`,
  runs the new contract/unit suites. (Default CI never installs the
  heavy extras.)
- Update `[tool.coverage.run].omit` only for *extra-only* modules
  (Gemini/VLA backends), keep pure-Python siblings (safety chain,
  pose PnP, configs) fully covered.

### X5 — Telemetry constants

Many `SPAN_*` constants are declared but never emitted. As each backend
lands, *emit* its span and pin the string in
`tests/unit/telemetry/test_span_constants.py`. Add new constants
(`SPAN_AGENT_EVAL`) in `armdroid.telemetry` only — never inline.

---

## Dependency-ordered execution

```
Phase 1 (unblock; parallel):   A0 ── (GPU)        B0 ── X3 (fakes) ── (camera)
Phase 2 (capability):          A1 → A2            B1, B2, B3   (Gemini perception/replan/VLA)
Phase 3 (safety + breadth):    A3, A4, A5         B4 (safety, + security-review)
Phase 4 (converge):            B6 (real loop) → C0 (sim-to-real) → B5 (Live API)
Phase 5 (cleanup / v0.4):      F5 (tighten step return type), F6 (remove mock_hardware)
```

- **A0 gates the entire Isaac track.** **B0 + X3 + B4 gate the real
  robot.** **C0 needs A2 + A4 + B4.**
- Each task: `Plan` agent design → implement on branch → `/code-review`
  (+ `/security-review` for B4/B6/transport) → green gates → commit
  (conventional, one behavioural diff) → gap-analysis pass.

## Per-task definition of done

1. Code is config-driven (no new literals), registry-registered (no
   `if kind ==`), backward compatible (default path byte-identical).
2. Structured logging + telemetry span on every external call.
3. Tests in the correct tier (unit/property/regression/contract/
   integration/e2e/isaac/hardware) using shared fakes; new contract test
   for any new backend; coverage ≥ 85% (currently ~96%).
4. Three gates green; `/code-review` clean.
5. Gap-analysis pass: no dead code, no duplication, no stale comment, no
   orphan config field / unused entry point.
6. Docs updated (ADR for protocol changes; example YAML for new config;
   `NEXT_STEPS.md` follow-up ID closed).
