# ADR-0005: Isaac Sim 5.1 / Isaac Lab 2.3 Backend

## Status

Accepted (PR-B, 2026-05-09).

## Context

The SO-ARM100/101 path needs a sim-to-real RL training pipeline. MuJoCo
is in place (PR-A) but lacks the photorealistic perception fidelity and
massively-parallel rollout that NVIDIA Isaac Lab 2.3 provides on top of
Isaac Sim 5.1. The LycheeAI [`MuammerBay/isaac_so_arm101`](https://github.com/MuammerBay/isaac_so_arm101)
reference project (BSD-3-Clause) already tunes the STS3215 servo PD
gains and provides a working `ManagerBasedRLEnv` reach task that
integrates with RSL-RL's `OnPolicyRunner`.

PR-A scaffolded the seams (`arm_driver_kind` discriminator,
`ArmRLAgentProtocol`, env+agent telemetry spans, vendored SO-ARM101
assets). PR-B lands the actual Isaac runtime code behind the optional
`armdroid[isaac]` extra.

## Decisions

1. **Opt-in `[isaac]` extras pattern.** Default `pip install -e ".[dev]"`
   never pulls isaaclab. The `[isaac]` extra pins
   `isaaclab[all,isaacsim]==2.3.0` and `rsl-rl-lib>=2.0`. Install
   requires `--extra-index-url https://pypi.nvidia.com` because PyPI
   does not host `isaacsim`. Documented in README, this ADR, and
   `gpu-ci.yml` (which sets `PIP_EXTRA_INDEX_URL`).

2. **`IsaacSimDriver` uses singular `set_joint_position_target`.** Isaac
   Lab 2.3's `isaaclab.assets.Articulation` API uses the singular form
   per [the docs](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html).
   The plural form (`set_joint_position_targets`) is on the legacy
   `omni.isaac.core.articulations.Articulation` (Isaac Sim Core API)
   which armdroid does NOT use. PR #8's research-doc review cited the
   plural form per gemini's reading of the legacy API; PR-B's
   verification (Phase 1 Explore agent against current 2.3 docs)
   confirms singular for the Lab path.

3. **`_TensorAdapter` enforces `num_envs == 1` for the protocol path.**
   `ArmEnvironmentProtocol.step()` returns scalar Python `float` /
   `bool`; Isaac Lab's `ManagerBasedRLEnv.step()` returns
   `torch.Tensor` shaped `(num_envs,)`. The adapter unwraps single-env
   tensors via `.reshape(-1)[0]` (defensive against `(num_envs, 1)`
   shape ambiguity). `num_envs > 1` raises `RuntimeError` pointing at
   the bypass path: `SoArmReachIsaacEnv._isaac_env`. Vectorised
   training (RSL-RL `OnPolicyRunner`) uses the bypass via
   `RslRlPpoAgent.build(env)` reaching through `env._isaac_env`.

4. **Gripper convention preserved.** armdroid is `0=open, 1=closed`
   per `domain/state.py` and `firmware/arm_esp32/PROTOCOL.md`. The URDF
   convention is reversed (positive radians = open jaw). Conversion
   lives in a single place: `armdroid.hardware.isaac_sim.gripper`,
   parametrised on `ArmSimIsaacConfig.{gripper_joint_radians_open,
   gripper_joint_radians_closed, gripper_normalised_open,
   gripper_normalised_closed, gripper_joint_index}`. Both
   `IsaacSimDriver` and `SoArmReachIsaacEnv` import from there.
   Closes R5 + R7 from PR #8 review.

5. **`RslRlPpoAgent` instantiation lives in `build_arm_controller`'s
   explicit branch, NOT the registry-dispatched path.** The
   registry's `Callable[..., ArmRLAgentProtocol]` factory generic
   cannot accommodate dual-config (`training_cfg` + `ppo_cfg`)
   without re-reading `ArmSettings()` inside the factory — which would
   silently drop YAML overlays (peer-review C-1). The registry
   registration of `rsl_rl_ppo` is a placeholder so
   `test_entry_point_mirror` passes; the orchestrator's
   `build_arm_controller` checks
   `if cfg.arm_training.algorithm == "rsl_rl_ppo":` and instantiates
   `RslRlPpoAgent(ppo_cfg=cfg.arm_rsl_rl_ppo, training_cfg=cfg.arm_training)`
   directly. SAC stays on the registry-dispatched path; symmetry is
   sacrificed for correctness.

6. **Vendored task code carries BSD-3 attribution.**
   `armdroid.environments.isaac.tasks.reach.*` is vendored from
   `MuammerBay/isaac_so_arm101 @ e4624dea075b00a36dbc66bebd531d191c92e8cd`.
   `armdroid.hardware.isaac_sim.articulation` is vendored from the
   same upstream's `robots/trs_so100/so_arm100.py`. All BSD-3 +
   Isaac Lab Project Developers attribution headers are preserved
   verbatim. Mechanical import-rewrites
   (`isaac_so_arm101.tasks.reach` → `armdroid.environments.isaac.tasks.reach`,
   etc.) applied via sed; no hand-edits to vendored logic.
   `THIRD_PARTY_NOTICES.md` indexes the chain. Ruff per-file-ignores
   for vendored task code cover upstream's coding style without
   rewriting BSD-3-licensed code.

7. **AppLauncher is a process singleton.** Isaac Sim's Kit cannot be
   launched twice in the same Python process. `IsaacSimDriver`
   enforces this via a module-level `_app_launched: bool` guard;
   second `connect()` raises `ArmDriverError` with a clear message
   rather than producing inscrutable Kit double-init traceback
   (peer-review N-3). Tests under `tests/isaac/` run sequentially
   (`pytest -n0`); default `make test` excludes the `isaac` marker so
   `pytest -xdist` parallelism is unaffected.

8. **`[isaac]` extra omits torch / torchvision pins.** torch is in
   armdroid's base `[project.dependencies]`; isaaclab's transitive
   resolution handles its torch needs. PR #8's
   `torch==2.7.0` / `torchvision==0.22.0` speculative pins (R2) are
   not reintroduced.

9. **USD output is a build artefact, not vendored.** Isaac Sim's URDF
   importer writes `assets/so_arm/so101/usd/so101.usd` on first call.
   `.gitignore` excludes `assets/**/usd/*.usd*` (PR-A B.18 placeholder).
   The vendored URDF + STL meshes (PR-A) are the source-of-truth.

10. **GPU CI is manual-dispatch + path-triggered, ubuntu-latest only.**
    `gpu-ci.yml` runs on `ubuntu-latest` (no self-hosted GPU runner).
    It installs `[dev,isaac]` (~8-10 GB; runner has ~55 GB free),
    asserts the import surface, and runs the pure-Python isaac-extra
    unit tests (gripper math, tensor adapter, configs). The actual
    GPU-bound smoke suite (`tests/isaac/`) only runs locally. The
    workflow trigger uses `paths` filtering with default event types
    AND a `gpu` label opt-in for cross-cutting PRs (peer-review
    hole-finder fix: `paths` is ignored under `types: [labeled]`).

## Consequences

* **Default install unchanged.** `pip install -e ".[dev]"` works as
  before. The `[isaac]` extra is opt-in.
* **Default CI matrix unchanged.** `make check` runs the same
  `pytest -m "not hardware and not isaac and not gpu"` lineup; coverage
  gate stays at 85% (Isaac modules are in `[tool.coverage.run].omit`).
* **Backwards compatibility preserved.** The legacy `mock_hardware:
  bool` field stays through v0.3.x with deprecation warning; removed
  in v0.4.0. `ArmTaskConfig.task_type`,
  `ArmTrainingConfig.algorithm`, and `arm_driver_kind` Literals widen
  additively.
* **AppLauncher singleton means `tests/isaac/*` runs sequentially.**
  Default `make test` excludes the `isaac` marker so xdist is unaffected.
* **Vectorised training bypasses the protocol** via
  `SoArmReachIsaacEnv._isaac_env`. F1 (a `VecArmEnvironmentProtocol`
  in v0.4) is the long-term resolution.

## Alternatives considered

* **Drake / PyBullet.** Neither has the photorealistic perception or
  GPU-massive rollout of Isaac Lab 2.3. Both deferred — the env
  registry seam admits future addition without touching the protocol.
* **ManiSkill3.** Viable; revisit if Isaac install footprint blocks
  contributors.
* **Skip vectorisation in protocol.** Accepted (decision 3). The
  `_TensorAdapter`'s `num_envs > 1` raise documents this.
* **Hardcode singular vs plural API at the call site.** Rejected —
  decision 2 records the verification chain (which API class, which
  docs URL, which version) so a future regression has a paper trail.

## See also

* PR-A merge: [PR #10](https://github.com/ianshank/Arm_Droid/pull/10)
* Implementation plan: PR-B issue / branch `claude/isaac-runtime-pr-b`
* Verification chain (Isaac Lab 2.3 API):
  <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html>
* MuammerBay reference: <https://github.com/MuammerBay/isaac_so_arm101>
* Pinned upstream commits: `THIRD_PARTY_NOTICES.md`
