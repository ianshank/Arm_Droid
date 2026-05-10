# Next Steps

This document tracks the roadmap from PR-A → PR-B → Phase 1.0 multi-
transport → tech-debt cleanup, plus the follow-ups still outstanding.
The PR-#10 (PR-A) and PR-#11 (PR-B) histories on GitHub are the
canonical record of what landed in the Isaac integration line.
``feature/phase-1.0-esp32-multi-transport`` is the parallel ESP32
multi-transport line; ``feature/tech-debt-cleanup`` is the current
follow-up that lands TD-1 / TD-4 / TD-5 / TD-6 / TD-7 cleanups on top
of the rebased Phase 1.0 base.

## Phase 1.0 + tech-debt cleanup (current branch)

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

## Follow-ups (post PR-B)

| ID | Description | Notes |
| --- | --- | --- |
| **F1** | Vectorised env path (`num_envs > 1`) bypassing `ArmEnvironmentProtocol`. | New `VecArmEnvironmentProtocol` in v0.4. The `_TensorAdapter` already raises `RuntimeError` with an informative message when `num_envs > 1`. |
| **F2** | GR00T / foundation-model integration. | Documented in PHASES.md Stretch axis. |
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
