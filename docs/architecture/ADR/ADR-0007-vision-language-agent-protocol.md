# ADR-0007: Vision-Language Agent Protocol (Phase A of Gemini ER 1.6)

## Status

Proposed (landing in Phase A of branch
`claude/gemini-robotics-embodied-reasoning-KrlEw`).

## Context

The Gemini ER 1.6 integration plan introduces a Vision-Language-Action
(VLA) control surface: a controller that consumes `(rgb, depth,
instruction, state)` and emits an `ArmAction` directly, with no
symbolic state extraction or RL training loop in between. The existing
`ArmRLAgentProtocol` cannot host this surface because its
`predict(observation: dict[str, NDArray[np.float64]])` signature is
fundamentally numeric-state-driven; pixels never enter the contract.

Two designs were considered:

1. **Widen `ArmRLAgentProtocol`** to accept either numeric or pixel
   observations via a union type. Rejected because every existing
   implementation (`SACAgent`, `RslRlPpoAgent`) would have to grow a
   discriminating branch, and the structural typing benefits of
   `@runtime_checkable` would be diluted.
2. **Add a sibling protocol** alongside `ArmRLAgentProtocol`, just as
   ADR-0006 added `VecArmRLAgentProtocol` next to it. Accepted.

The peer review of the v1 integration plan caught a correctness
hazard in option 2 (BLOCKER B3): `@runtime_checkable Protocol` matches
by **method-name set only**, so if `VisionLanguageAgentProtocol` and
`ArmRLAgentProtocol` share any method names (e.g. `build`,
`is_built`), then `ArmController` cannot dispatch via two `isinstance`
checks without ambiguity. ADR-0006 sidestepped this by giving
`VecArmRLAgentProtocol` deliberately disjoint names (`build_vec`,
`train_vec`). This ADR follows the same discipline.

## Decision

1. **Sibling protocol with disjoint method names.** Introduce
   `VisionLanguageAgentProtocol` in `armdroid.domain.protocols` with
   methods `build_vla`, `act`, `is_vla_built`, `is_vla_loaded`. None
   of these collide with `ArmRLAgentProtocol`'s methods (`build`,
   `train`, `predict`, `save`, `load`, `is_built`, `is_trained`). The
   structural disjointness is pinned by
   `tests/unit/domain/test_new_protocols.py::test_rl_agent_does_not_match_vla_protocol`.
2. **`act` boundary type is `ArmAction`.** A new frozen, slotted
   dataclass in `armdroid.domain.state` carrying
   `joint_targets: tuple[float, ...]`, `gripper: float`, optional
   `timestamp_s: float | None`. Equality + hashing are structural,
   matching the existing `ArmState` convention. `from_array` provides
   a boundary converter for backends that produce numpy arrays.
3. **Dispatch in `ArmController` via `isinstance`.** Phase D
   `ArmController.__init__` accepts either an `ArmRLAgentProtocol`
   instance or a `VisionLanguageAgentProtocol` instance and stores the
   matched-protocol flag for routing. With disjoint methods the
   dispatch is unambiguous; a `SACAgent` instance fails the VLA
   `isinstance` check at construction time.
4. **Registry mirror.** Phase D adds the
   `armdroid.vision_agents` entry-point group and a `Registry[type[
   VisionLanguageAgentProtocol]]` under `armdroid.control.vla.registry`.
   The registry follows the same shape as
   `armdroid.control.registry._RL_AGENTS`.
5. **Opt-in via config.** `ArmVlaConfig.enabled` defaults to `False`;
   `build_arm_controller` consults the flag and falls through to the
   existing SAC / PPO path otherwise. No legacy training surface
   changes.

## Consequences

* **Positive.** Pixels-in / actions-out is a clean separate axis. Both
  the RL and VLA surfaces evolve independently. Existing tests pass
  unchanged. Phase D regression tests pin the dispatch correctness.
* **Negative.** `ArmController` grows a second injection slot;
  documentation must explain the routing. The `isinstance` dispatch
  costs a single attribute lookup per construction (negligible).
* **Risks.** If a future protocol introduces a method name that
  collides with either family, the runtime dispatch becomes
  ambiguous. Mitigation: every new sibling protocol must satisfy a
  contract test asserting disjoint method names.

## Alternatives considered

* **Subclass `ArmController`** with a `VlaArmController` variant.
  Rejected because the orchestrator already constructs `ArmController`
  by name; an inheritance hierarchy would require a separate factory
  branch and lose the protocol-driven uniformity.
* **String discriminator on `ArmTrainingConfig.algorithm`.** Adding
  `"gemini_vla"` as an algorithm value was considered. Rejected
  because the VLA path has no `train` semantics; co-locating it under
  `arm_training` would muddy the boundary.

## References

* Peer-review blocker B3 (resolution table in plan §12 Appendix).
* ADR-0006: precedent for sibling-protocol design + disjoint method
  names.
* Plan section 3 Phase A + Phase D for the implementation timing.
