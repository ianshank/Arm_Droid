# Next Steps

This document tracks the post-PR-A roadmap for the Isaac Sim / Isaac Lab
integration plus the broader follow-ups surfaced during the PR-A scan.
The full plan lives at
[`C:\Users\iansh\.claude\plans\please-analyze-and-research-shiny-wozniak.md`](../.claude/plans/please-analyze-and-research-shiny-wozniak.md);
this file is the executive summary.

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
