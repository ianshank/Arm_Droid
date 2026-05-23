# ADR-0009: Swiss-cheese safety middleware (Phase A scaffolding)

## Status

Proposed (landing in Phase A of branch
`claude/gemini-robotics-embodied-reasoning-KrlEw`; concrete guards in
Phase F).

## Context

Adding an autonomous "brain" (Gemini ER 1.6) to heavy robot hardware
demands layered safeguards. The DeepMind talk that motivated this
work proposes a "Swiss-cheese" defence model with deny-overrides
semantics: any single layer's denial short-circuits the chain. The
constituent layers in Phase F are:

* **ISO TS 15066-derived kinematic guard** - velocity, force, joint
  velocity caps from `ArmSafetyConfig`.
* **Asimov natural-language guard** - Gemini ER adjudication when
  available, deterministic-ruleset fallback otherwise.
* **Fragile-object guard** - consumes `DetectedObject.is_fragile` /
  `affordances`.

The orchestrator and the controller both need to consult the chain:
the orchestrator before dispatching a planned step, the controller
before sending an action to the driver.

The peer review surfaced one packaging gap (M3): the NEISS-derived
corpus the Asimov guard consumes must ship inside the wheel so
operators do not have to download it at install time. Phase A
reserves the wheel asset slot; Phase F populates it.

## Decision

1. **`SafetyGuardProtocol`** lives in `armdroid.domain.protocols` from
   Phase A. Methods: `check_plan(steps) -> Verdict`,
   `check_action(action, state) -> Verdict`, and a `name` property
   for logs.
2. **`Verdict`** is a frozen, slotted dataclass in
   `armdroid.domain.state`: `allowed: bool`, `reason: str = ""`,
   `guard_name: str = ""`. Permissive default keeps construction
   ergonomic.
3. **Deny-overrides chain composer** lands in
   `armdroid.safety.swiss_cheese` (Phase F). A
   `Registry[Callable[[ArmSafetyConfig], SafetyGuardProtocol]]`
   under entry-point group `armdroid.safety_guards` supplies the
   built-in guards.
4. **Default chain is empty.** `cfg.arm_safety.asimov_rule_set` is
   `"off"` by default and `build_safety_chain(cfg)` returns `[]`.
   Existing rollouts see no behavioural change.
5. **NEISS corpus packaging.** `pyproject.toml` Phase A declares a
   `[tool.hatch.build.targets.wheel.force-include]` mapping
   `assets/safety` to `armdroid/safety/data` so the JSON corpus
   ships inside the wheel. The corpus path is configurable
   (`ArmSafetyConfig.neiss_corpus_path`) for operator overrides.
   Licensing review of the redistributed subset is open question 3
   in the plan.
6. **Guard invocation telemetry.** SPANs
   `SPAN_SAFETY_CHECK_PLAN`, `SPAN_SAFETY_CHECK_ACTION`, and
   `SPAN_SAFETY_DENY` are declared in Phase A so all guards emit a
   uniform observability surface.

## Consequences

* **Positive.** A single, registry-driven chain replaces ad-hoc
  driver-level safety checks. Default chain is empty so legacy
  behaviour is identical until an operator opts in.
* **Negative.** Two more cross-cutting concerns added to the
  orchestrator's hot path (`check_plan` before dispatch,
  `check_action` before driver send). Phase F benchmarks must
  measure latency cost; the empty-chain default is a no-op.
* **Risks.** Adding the chain risks blocking legitimate motion
  during operator configuration (plan §11 R-G5). Mitigation: the
  regression test
  `tests/regression/test_default_safety_passthrough.py` pins the
  empty-chain pass-through.

## Alternatives considered

* **Push safety into the driver.** Rejected because the driver
  cannot reason about plan-level concerns (NEISS injury categories,
  goal text). A guard chain that wraps planner output is a strictly
  more capable surface.
* **Hardcoded ISO velocity caps in the controller.** Rejected on
  the no-hardcoded-values rule. Operators must be able to tune
  caps per OEM via Pydantic config.

## References

* Peer-review MEDIUM M3 (NEISS packaging).
* Plan section 3 Phase F for guard implementations.
* Plan section 11 R-G5 and R-G8 risks.
