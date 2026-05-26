# Next Steps

This document tracks the roadmap, milestones, and outstanding/landed items for the `armdroid` platform.

## Landed: Physical Production Blocker Resolution (current branch)

We have successfully resolved all major hurdles blocking physical robot arm deployment in the real world (Sim-to-Real):

* **Physical Calibration Utility** (`scripts/calibrate_arm.py`) — Provides an interactive command-line interface for human operators to step joints to extreme positions to determine min/max rad boundaries and home poses safely, exporting local overlays to `config/calibrated_limits.yaml`.
* **Domain Randomization Bridge** (`src/armdroid/environments/isaac/_domain_randomization.py`) — Integrated physical scaling ranges (mass, friction multipliers) and joint PD stiffness/damping gain noise directly into the environment reset hooks using Pydantic configurations on `ArmSimIsaacConfig` to enable robust zero-shot Sim-to-Real transfer.
* **OpenCV PnP Pose Estimator** (`src/armdroid/perception/pose_estimator.py`) — Replaced the placeholder zero orientation with a real OpenCV Perspective-n-Point (`solvePnP`) 6-DoF pose estimator utilizing 2D detection keypoints, 3D object geometries, and camera/distortion matrices.
* **NVIDIA Jetson Containerization** (`Dockerfile`, `docker-compose.jetson.yml`, `.dockerignore`) — Multi-stage edge build optimized for Linux for Tegra (L4T PyTorch base) with device passthrough rules for ESP32 UART serial and RealSense camera feeds, along with Bluetooth dbus sharing.
* **Diagnostics Health Probe** (`scripts/jetson_health_check.py`) — Container health probe verifying CUDA availability, depth camera feed, and ESP32 UART port status.
* **Regression and Type Hardening** (`tests/regression/test_sim2real_perception_regression.py`) — Added regression checks ensuring validation bounds of domain randomization ranges and object geometry specifications.

## Phase 1.0 + tech-debt cleanup (previous branch milestones)

Landing in ``feature/tech-debt-cleanup``:

* **TD-1 (HIGH)** — `mypy>=1.13` pin; the prior `>=1.5` allowed 1.11.x
  which crashes in strict mode on `importlib.util.find_spec` callers
  (the new probe + `telemetry/otel.py` + `ble_transport.py` all use
  that pattern). `mypy --strict src/` is once again clean across 105
  source files.
* **TD-4 (MEDIUM)** — `SACAgent.build` signature tightened from
  `env: Any` to `env: ArmEnvironmentProtocol`; SB3 boundary handled
  by an explicit `cast` to `gym.Env`. Runtime conformance unchanged
  (still `isinstance(SACAgent, ArmRLAgentProtocol) is True`).
* **TD-5 (MEDIUM)** — `ArmOrchestrator` gains an optional
  `task_cfg: ArmTaskConfig | None = None`; rollout's
  `_resolve_target_position(args, task_cfg)` walks PDDL plan-step args
  right-to-left (planner convention puts destination last) and looks
  up real Cartesian XYZ from `task_cfg.peg_positions` /
  `basket_positions` instead of always returning zeros. Backward-
  compat: legacy callers without `task_cfg` keep working with a
  one-time structured warning per rollout.
* **TD-6 (deferred)** — `BaseEnv.render` is documentation-only for
  now; real implementation needs MJCF loading + an offscreen
  framebuffer pipeline (`mujoco.Renderer`). Tracked separately because
  the base env is a numpy state model with no MuJoCo backing; wiring
  it in is scope-orthogonal to the rest of this branch.
* **TD-7 (deferred)** — `PoseEstimator` orientation stays zero until
  OpenCV PnP is wired up; needs (a) per-class 3D model points
  extracted from URDF / mesh, (b) corresponding 2D detections beyond
  the current bbox center, and (c) a calibrated camera matrix.
  Documented at the call site with `orientation_unmeasured=True` in
  the structured log so consumers can detect the placeholder.
* **TD-8 (audit-resolved)** — the lazy-build factory pattern
  (`controller.build_for_env(env)` lazy on first train) was confirmed
  intentional. The pre-rebase "RL agent classes do not yet share a
  single Protocol" comment is now stale (PR-A added
  `ArmRLAgentProtocol`); refreshed the registry docstring to reflect
  the post-rebase reality.

Plus the supporting work that landed alongside:

* `scripts/check_isaac_install.py` — cross-platform pre-install probe
  for the `[isaac]` extra; safe on any host; reports Python /
  driver / per-GPU compute / package state with a name-based
  heuristic fallback for mixed Tesla / GeForce driver layouts.
* ESP32 Phase 1.0 multi-transport — `armdroid.hardware.esp32.transport`
  package with `SerialTransport`, `TcpTransport`, and `BleTransport`,
  plus `TransportAuthConfig` (HMAC-SHA256). The `[ble]` extra adds
  `bleak>=0.22`. Worked YAML examples in `config/examples/`.

## Original PR-A → PR-B roadmap

> **Background note:** the original research doc
> `docs/research/isaac-omniverse-integration.md` was merged via PR #8
> and subsequently reverted on `main` via PR #9. PR-A keeps the
> reverted state (the file does not return). PR-B's implementation
> stands on its own — `ArmSimIsaacConfig` is the spec for the gripper
> conversion knobs and servo PD gains; the `_TensorAdapter` design
> recorded below replaces the reverted §6.1 snippet.

## Immediate next PR — PR-B (Isaac integration behind `[isaac]` extra)

PR-A landed every backwards-compatible refactor PR-B needs. PR-B drops
the actual Isaac runtime code behind the optional `armdroid[isaac]`
extra. **Default `pip install -e ".[dev]"` and the default CI matrix
remain unaffected.**

### Modules PR-B introduces

| Module | Responsibility |
| --- | --- |
| `armdroid.config.schema.sim_isaac` | `ArmSimIsaacConfig` — every numeric servo PD gain, init-pose joint, env id, joint name, gripper conversion knob is a Pydantic field with `ge`/`le`/`gt` bounds (no hardcoded values). |
| `armdroid.config.schema.training` (extension) | `RslRlPpoConfig` mirroring `MuammerBay/isaac_so_arm101` `ReachPPORunnerCfg` defaults pinned at the upstream commit SHA in `assets/so_arm/so101/ATTRIBUTION.md`. |
| `armdroid.hardware.isaac_sim.gripper` | Pure conversion functions `normalised_to_radians` / `radians_to_normalised` parametrised on `ArmSimIsaacConfig`. Single source of truth for the rescale-and-invert that closes R5 + R7. Tested without isaaclab. |
| `armdroid.hardware.isaac_sim.articulation` | Vendored `MuammerBay/isaac_so_arm101` `ArticulationCfg` builder reading every PD gain from config. BSD-3-Clause attribution preserved. |
| `armdroid.hardware.isaac_sim.driver` | `IsaacSimDriver` satisfying `ArmDriverProtocol`. Lazy isaaclab + torch imports inside `connect()`. Uses **plural** `set_joint_position_targets` (R4). Telemetry spans + structlog events identical to `Esp32JsonDriver`. |
| `armdroid.environments.isaac.common._TensorAdapter` | Bridges Isaac Lab `ManagerBasedRLEnv` tensor returns ↔ `ArmEnvironmentProtocol` numpy returns. Asserts `num_envs==1` for the protocol path. Defensive `np.asarray(reward).reshape(-1)[0]` guard against `(num_envs, 1)` shape. Tests use real torch (already in base deps) so the production `t.cpu().numpy()` path is exercised. |
| `armdroid.environments.isaac.reach.SoArmReachIsaacEnv` | `ArmEnvironmentProtocol`-conforming wrapper. |
| `armdroid.environments.isaac.tasks.reach.*` | Vendored from `MuammerBay/isaac_so_arm101/tasks/reach/` (BSD-3-Clause, attribution headers preserved). |
| `armdroid.control.rsl_rl_agent.RslRlPpoAgent` | `ArmRLAgentProtocol`-conforming wrapper around `rsl_rl.runners.OnPolicyRunner`. Lazy `from rsl_rl import …` inside `build()`. Uses the uniform `from_settings(cls, ArmSettings)` classmethod (closes F7 from the plan — eliminates the `# type: ignore[call-arg]` in `factory.py`). |

### Pyproject changes PR-B applies

```toml
[project.optional-dependencies]
isaac = [
    "isaaclab[all,isaacsim]==2.3.0",
    "rsl-rl-lib>=2.0",
    # NO speculative torch/torchvision pin (R2): isaaclab's transitive
    # resolution is the source of truth.
]

[project.entry-points."armdroid.drivers"]
isaac_sim = "armdroid.hardware.isaac_sim:IsaacSimDriver"

[project.entry-points."armdroid.environments"]
so_arm_reach_isaac = "armdroid.environments.isaac.reach:SoArmReachIsaacEnv"

[project.entry-points."armdroid.rl_agents"]
rsl_rl_ppo = "armdroid.control.rsl_rl_agent:RslRlPpoAgent"
```

PR-B also widens `ArmSettings.arm_driver_kind` to
`Literal["mock", "esp32", "isaac_sim"] | None`,
`ArmTrainingConfig.algorithm` to add `"rsl_rl_ppo"`,
and `ArmTaskConfig.task_type` to add `"so_arm_reach_isaac"`.

### CI

PR-B adds `.github/workflows/gpu-ci.yml` triggered by `workflow_dispatch`
or a `gpu` PR label. Per the user's directive (no self-hosted runner),
the workflow installs `[dev,isaac]` on `ubuntu-latest`, asserts the
`[isaac]` import surface compiles cleanly, and runs the pure-Python
isaac-extra unit tests. The full GPU smoke suite
(`tests/isaac/test_smoke_isaac_driver.py`) only runs locally with
`ARMDROID_ISAAC_RUN=1 pytest tests/isaac`. If `pip install isaaclab`
exceeds runner disk quota in practice, the workflow falls back to
`rsl-rl-lib` only and documents the dropped coverage.

### Backward compatibility checkpoints PR-B respects

* Every `ArmSettings(mock_hardware=…)` construction in tests continues to work.
* The legacy bool path keeps emitting one `DeprecationWarning` per process
  (already exercised in `tests/integration/hardware/test_factory_dispatch.py`).
* `mock_hardware` is removed in **v0.4.0**, not v0.3.x. PR-B does not
  delete it.
* `make check` gate stays green: Isaac modules live in the coverage omit
  list; pure-Python siblings (gripper math, configs) are explicitly
  covered.

## Landed in `claude/gemini-robotics-embodied-reasoning-KrlEw`

* **Phase A of Gemini Robotics ER 1.6 integration** — landed via
  ADRs 0007 / 0008 / 0009. Foundation scaffolding only; zero
  behavioural change to default rollouts. Adds:
  * Four new `@runtime_checkable` protocols in
    `armdroid.domain.protocols`: `VisionLanguageAgentProtocol`
    (with disjoint method names from `ArmRLAgentProtocol` per
    ADR-0007 to keep the Phase D `isinstance` dispatch
    unambiguous), `HighLevelPlannerProtocol`,
    `SafetyGuardProtocol`, `InteractionSessionProtocol`.
  * Four new frozen, slotted value objects in
    `armdroid.domain.state`: `ArmAction` (with `from_array`
    boundary converter), `SceneInsight`, `Verdict`,
    `InteractionEvent`. `DetectedObject` widened with five
    keyword-only fields preserving the positional six-arg ctor;
    `__slots__` pinned by regression test.
  * Three new sub-configs (`ArmVlaConfig`, `ArmSafetyConfig`,
    `ArmInteractionConfig`) wired into `ArmSettings`.
    `LLMReplannerConfig.backend` widened to include `"gemini"`.
    `ArmPerceptionConfig` gains `detector_kind`, frame-buffer,
    open-vocab, and Gemini-envelope fields. All numeric bounds
    enforced.
  * 17 new module-level `SPAN_*` constants in
    `armdroid.telemetry` for perception / replanner / VLA /
    interaction / safety families. String values pinned by
    `tests/unit/telemetry/test_span_constants.py`.
  * Three new opt-in extras in `pyproject.toml`: `[gemini]`,
    `[gemini-live]`, `[gemini-vla]`. Wheel asset slot reserved
    for Phase F's NEISS safety corpus.
  * `tests/helpers/fake_llm_sdk` lifted from
    `tests/unit/planning/test_llm_replanners.py` so Phases B-E
    test suites share one provider-agnostic fake SDK harness.
  * `config/examples/gemini.example.yaml` exhaustive overlay,
    documented + regression-tested.
  * Three ADRs (0007 / 0008 / 0009) plus two extension guides
    (`add-a-vision-agent.md`, `add-a-detector.md`).
  * Public-API re-exports lifted into `armdroid`, `armdroid.api`,
    and `armdroid.domain` so the new types and protocols are
    discoverable through every documented import path; pinned by
    identity-checked regression at
    `tests/regression/test_public_api_phase_a.py`.

  Quality: 910 tests pass (+66 net), coverage 95.53%, ruff +
  mypy `--strict` clean. See
  `docs/architecture/ADR/ADR-0007-vision-language-agent-protocol.md`
  (and 0008 / 0009) plus the integration plan at
  `/root/.claude/plans/as-per-the-research-radiant-nova.md`.

## Landed in `feature/f1-vec-env-protocol`

* **F1 (Vectorised env path, `num_envs > 1`)** — landed via ADR-0006.
  Introduces `VecArmEnvironmentProtocol` and `VecArmRLAgentProtocol`
  as siblings to the existing protocols (not generic parameters).
  `SoArmReachIsaacVecEnv` owns `ManagerBasedRLEnv` directly and
  coordinates with `armdroid.hardware.isaac_sim._app_state` so Kit is
  never double-booted. `RslRlPpoAgent.build_vec` / `train_vec` replace
  the `env._isaac_env` reach-through cleanly via
  `VecArmEnvironmentProtocol.as_runner_env()`. The factory routes on
  `cfg.arm_sim_isaac.num_envs > 1` via `_VEC_CAPABLE_ALGORITHMS` and
  `_VEC_TASK_REGISTRY_NAMES` allow-list mapping. Single-env path is byte-identical.
  See [docs/architecture/ADR/ADR-0006-vec-env-protocol.md](docs/architecture/ADR/ADR-0006-vec-env-protocol.md).

## Follow-ups (post PR-B)

| ID | Description | Notes |
| --- | --- | --- |
| **F2** | Foundation-model integration. Phase A scaffolding landed via ADRs 0007-0009 on `claude/gemini-robotics-embodied-reasoning-KrlEw`. Phases B (perception detector + frame buffer + scene reasoner), C (`GeminiReplanner` + LLM-replanner registry + factory wiring), D (`VisionLanguageAgentProtocol` + Gemini VLA backend), E (Live API), F (Asimov safety chain) are the remaining work. | See `/root/.claude/plans/as-per-the-research-radiant-nova.md`. |
| **SAF-1** | Safety roadmap follow-up reserved by ADR-0009. NEISS corpus refresh cadence, OEM-specific velocity profiles, and operator-supplied custom rule sets. Lands alongside Phase F. | New track. |
| **F3** | Real-time Isaac Sim GUI viewer. | Currently headless only. |
| **F4** | Domain-randomisation parity between MuJoCo and Isaac. | Mass / friction / lighting ranges currently MuJoCo-only. |
| **F5** | Tighten `ArmEnvironmentProtocol.step` return signature from `dict[str, Any]` to `dict[str, NDArray[np.float64]]`. | Touches every env. Defer to a dedicated protocol-tightening PR. |
| **F6** | Remove `mock_hardware` field in v0.4.0 once the deprecation cycle completes. | Update `tests/conftest.py`, `mock_settings` fixture, every direct `ArmSettings(mock_hardware=…)` construction in tests in the same PR. |

## Risks documented for PR-B

* **R-1** — Isaac Sim 5.1 + Isaac Lab 2.3 + Python 3.11 version triple
  may drift. Mitigation: pinned in `[isaac]` extras; smoke test on
  `workflow_dispatch` catches drift early.
* **R-2** — Vendoring BSD-3 code into the MIT repo requires preserved
  copyright headers; `THIRD_PARTY_NOTICES.md` indexes vendored content.
* **R-3** — The `_TensorAdapter` `num_envs > 1` raise behaviour means
  the protocol path is single-env only. Acceptable per F1.
* **R-4** — `isaaclab` is a ~9 GB install. Default CI never installs
  it; only contributors with a GPU run the full Isaac suite.
* **R-5** — AppLauncher is process-singleton; `tests/isaac/*` runs
  sequentially (`pytest -n0`). Default `make test` excludes the
  `isaac` marker so xdist parallelism is unaffected.
* **R-6** — Confirming Isaac 5.1 API: `set_joint_position_targets`
  plural correct, reward tensor shape `(num_envs,)` not
  `(num_envs, 1)`, STL meshes accepted by URDF importer without
  manual convex decomposition. These can ONLY be confirmed on a real
  GPU box during PR-B local smoke testing.

## What PR-A explicitly did NOT do

* **Did not** add a runtime dependency. `pip install -e ".[dev]"` is
  unchanged.
* **Did not** widen the `arm_driver_kind` Literal beyond `["mock", "esp32"]`.
  PR-B widens it alongside the registry registration to keep the
  intermediate state crash-free.
* **Did not** remove `mock_hardware` — that's v0.4.0.
* **Did not** add ADR-0005. The plan drafts the content in §"ADR-0005
  draft"; PR-B writes it to `docs/architecture/ADR/ADR-0005-isaac-sim-backend.md`.
* **Did not** restore `docs/research/isaac-omniverse-integration.md`
  (reverted on `main` via PR #9). The R4 / R5 / R7 / R8 doc-level
  fixes that this branch had applied to that file are no longer
  visible in the diff; the implementation-level closures live in
  PR-B's `gripper.py`, `articulation.py`, `driver.py`, and
  `ArmSimIsaacConfig`.
