# ADR-0006: Vectorised environment protocol (F1)

## Status

Accepted (landed via `feature/f1-vec-env-protocol`).

## Context

`NEXT_STEPS.md` follow-up **F1** called for lifting the `num_envs == 1`
constraint baked into `_TensorAdapter` and `SoArmReachIsaacEnv` so
RSL-RL PPO can train against parallel Isaac Sim environments. The
previous workaround (`getattr(env, "_isaac_env", None) or env` in
`RslRlPpoAgent.build`) broke the protocol contract and was flagged as
scope-orthogonal in PR-B. The driver path
(`IsaacSimDriver`) already explicitly rejected `num_envs != 1` and
pointed users at a vec path that did not yet exist.

## Decision

1. **Sibling protocols, not a generalised generic.** Add
   `VecArmEnvironmentProtocol` next to `ArmEnvironmentProtocol` rather
   than parametrising the existing one. The return-type contracts
   differ structurally (torch tensors vs numpy scalars) and
   `@runtime_checkable` strips generic parameters at runtime, so a
   generic protocol would lose the dispatch discriminator. Same for
   `VecArmRLAgentProtocol` next to `ArmRLAgentProtocol`.
2. **Separate registry.** A new `_VEC_ENVIRONMENTS` registry under
   entry-point group `armdroid.vec_environments`. Mirrors the
   single-env `_ENVIRONMENTS`. A union registry would force every
   caller into an isinstance narrow before use.
3. **`as_runner_env()` accessor.** `VecArmEnvironmentProtocol` exposes
   a documented `as_runner_env()` method that returns the underlying
   `ManagerBasedRLEnv`. This replaces the `env._isaac_env`
   reach-through cleanly without subclassing the isaaclab type
   (subclassing would force `isaaclab` into the module-load critical
   path).
4. **Dispatch in `build_arm_environment` + `ArmController.build_for_env`.**
   The env-shape dispatch (single vs vec) happens at env construction;
   the agent-method dispatch (`build` vs `build_vec`) happens when the
   controller binds. Both axes are decoupled, both default to
   single-env when `num_envs == 1`. Module constants
   `_VEC_CAPABLE_ALGORITHMS` and `_VEC_CAPABLE_TASKS` pin the allow-list.
5. **`num_envs >= 1` permitted on the vec env.** The vec env accepts
   `num_envs == 1` as a degenerate case so callers may adopt the vec
   protocol unconditionally. Factory dispatch still routes
   `num_envs == 1` to the single-env path by default.
6. **AppLauncher singleton coordination.** `SoArmReachIsaacVecEnv`
   probes `armdroid.hardware.isaac_sim._app_state.is_app_launched()`
   before building and calls `mark_launched()` after a successful boot
   when it was the first booter. This prevents a later
   `IsaacSimDriver` construction in the same process from
   double-booting Kit (and vice versa).
7. **No deprecation for `env._isaac_env`.** The attribute is
   leading-underscore private; the only known internal caller
   (`RslRlPpoAgent.build` single-env path) is left unchanged for
   backwards-compat. A `DeprecationWarning` on a private attr would
   be theatre - removal will simply happen alongside other v0.4.0
   cleanups.
8. **No `num_envs` field in `ArmTrainingConfig`.** The seam stays in
   `ArmSimIsaacConfig.num_envs` where Pydantic already enforces
   `ge=1, le=16384`.

## Consequences

- Backwards-compatible: `num_envs == 1` (default) keeps the byte-identical
  single-env path. `_TensorAdapter` remains as the runtime seam. SAC stays
  single-env only. The single-env `build()` body on `RslRlPpoAgent` is
  literally untouched.
- The pure-Python `registry_vec.py` is covered by the 85% gate; the
  Isaac-runtime `reach_vec.py` is omitted via the existing
  `*/armdroid/environments/isaac/*` coverage wildcard.
- `ArmControllerProtocol.build_for_env` signature widened to
  `ArmEnvironmentProtocol | VecArmEnvironmentProtocol`. `ArmOrchestrator.environment`
  field similarly widened. Existing callers compile unchanged because
  they pass single-env types which still satisfy the left side of the
  union.
- Shared test invariants live in `tests/property/_vec_invariants.py`
  to prevent drift between single-env and vec env property suites.
- Six new telemetry SPAN constants added (`SPAN_AGENT_BUILD_VEC`,
  `SPAN_AGENT_TRAIN_VEC`, `SPAN_ENV_VEC_RESET`, `SPAN_ENV_VEC_STEP`,
  `SPAN_ENV_VEC_CLOSE`, `SPAN_ENV_VEC_KIT_BOOT`).

## Alternatives considered

- **Generalised `ArmEnvironmentProtocol[T]`** - rejected. Three type
  params (`T_obs`, `T_action`, `T_reward`) to honestly model the diff;
  `isinstance` loses the discriminator; `torch` would propagate
  through every call site instead of staying in one TYPE_CHECKING import.
- **Single registry with union value type** - rejected. Loses the
  `Registry[T].get` type discriminator; every caller needs an
  isinstance narrow.
- **Subclass `ManagerBasedRLEnv`** - rejected. Forces `isaaclab`
  import at module load (breaks the lazy-import discipline) and
  propagates Isaac Lab API drift directly into the project's
  stability surface.
- **Branch dispatch on `cfg.arm_training.algorithm`** - rejected.
  Breaks the moment a second vec-capable algorithm appears.
  `_should_use_vec(cfg) and algo in _VEC_CAPABLE_ALGORITHMS` is robust
  to future agents.
- **DeprecationWarning on `_isaac_env`** - rejected. A
  `DeprecationWarning` on a private underscored attribute is a
  contradiction; the attribute will be removed in v0.4.0 alongside
  other private-API cleanups, with no deprecation cycle.

## References

- `NEXT_STEPS.md` follow-up F1 (Vectorised env path).
- `src/armdroid/domain/protocols.py` - the new protocols.
- `src/armdroid/environments/registry_vec.py` - the new registry.
- `src/armdroid/environments/isaac/reach_vec.py` - the new env.
- `src/armdroid/control/rsl_rl_agent.py` - the new `build_vec` /
  `train_vec` methods + extracted `_iterations_for` helper.
- `src/armdroid/orchestration/factory.py` - `_should_use_vec` +
  `_VEC_CAPABLE_ALGORITHMS` / `_VEC_CAPABLE_TASKS`.
- `src/armdroid/control/controller.py` - protocol-type dispatch in
  `build_for_env` / `train_policy`.
