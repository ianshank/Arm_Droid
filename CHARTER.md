# Project Charter — armdroid (stable long-term plan)

> **Status: STABLE / SLOW-CHANGING.** This document is the north-star that humans and AI
> agents (and sub-agents) read first to stay on track. It changes rarely and only by
> deliberate decision — not per task. Day-to-day, changeable work lives in
> [`NEXT_STEPS.md`](NEXT_STEPS.md); the architecture-in-one-screen and contributor
> conventions live in [`AGENTS.md`](AGENTS.md); C4 detail in
> [`docs/architecture/C4.md`](docs/architecture/C4.md); the long-range axis roadmap in
> [`docs/architecture/PHASES.md`](docs/architecture/PHASES.md). If a task seems to
> require changing this charter, **stop and raise it explicitly** rather than editing it
> in passing.
>
> **Charter integrity is governed by §7 (Amendment Protocol).** Amendments are budgeted.
> When a budget or trigger fires, the amendment path closes and a full charter review is
> mandatory. Procedural tidiness is not a substitute for scope discipline.

---

## 0. Executive Summary

> armdroid is a hierarchical robot-arm training and control platform for makers and
> robotics researchers operating a MakerWorld 6-DoF arm driven over a JSON-over-UART
> protocol on an ESP32. It trains manipulation policies entirely in simulation (MuJoCo,
> and Isaac Sim/Lab behind an optional extra) with SAC+HER and PDDL planning, then
> transfers them to the physical arm over USB serial; it deliberately does **not** learn
> on the physical hardware and is **not** a general-purpose robotics middleware. The
> charter has 0 active carve-outs against a per-non-goal budget of 2; the last full
> charter review was never (initial ratification pending). Execution detail lives in
> NEXT_STEPS.md; this document defines intent, boundaries, and invariants only.

---

## 1. Vision

A maker or researcher with a low-cost 6-DoF arm can go from "bare hardware on the bench"
to "a policy that reliably manipulates objects" without ever risking the arm to learn:
they calibrate the physical joint limits once, train and validate entirely in
simulation, and transfer a robust policy to the real arm over a single serial cable.
The platform stays small, sharp, and well-tested, so that new hardware, simulators,
planners, and learning algorithms plug in through documented seams rather than forks of
the core.

## 2. Mission (what we are building)

- **Sim-to-real training stack** — train manipulation policies in MuJoCo (and Isaac
  Sim/Lab) with SAC+HER and transfer zero-shot to a physical arm via domain
  randomization and PnP-based perception.
- **Layered four-stage runtime** — perception, symbolic (PDDL) planning + LLM
  replanning, RL control, and hardware execution, each behind a `@runtime_checkable`
  Protocol wired at a single composition root.
- **Registry-backed extension seams** — drivers, environments, planners, and RL agents
  are added as plug-ins through entry-point groups, never by editing core factory code.
- **Physical hardware execution + calibration** — a JSON-over-UART ESP32 driver, an
  interactive joint-limit calibration utility, and an edge deployment path for NVIDIA
  Jetson with a health probe.
- **Stable, typed public surface** — a SemVer-versioned `armdroid` / `armdroid.api`
  facade with compatibility shims, strict typing, and a high coverage floor.

## 3. Scope

**In scope:** the `src/armdroid/` package tree (public API, orchestration, hardware /
planning / perception / control / environments adapters, config, domain protocols,
telemetry, compat shims); the ESP32 wire-protocol contract in `firmware/`; the training
and deployment scripts in `scripts/`; configuration schema and overlays in `config/`;
tests in `tests/`; and documentation in `docs/`.

**Out of scope (non-goals):** Each non-goal carries a stable ID. IDs are never reused or
renumbered; a retired non-goal keeps its ID with status `RETIRED`. Amendments (§7) are
counted against these IDs.

| ID | Non-goal | Status | Active carve-outs |
|------|----------|--------|-------------------|
| NG-1 | armdroid does not perform learning or policy optimization on the physical arm; the real hardware is used only for calibration and execution of policies trained in simulation. | ACTIVE | 0 / 2 |
| NG-2 | armdroid is not a general-purpose robotics middleware or framework (it is not a ROS/ROS 2 replacement); it targets the MakerWorld-class 6-DoF arm through the driver seam. | ACTIVE | 0 / 2 |
| NG-3 | armdroid core does not hard-depend on heavy or GPU-bound backends (Isaac, BLE, telemetry, CUDA); these remain opt-in extras that a default `pip install -e ".[dev]"` never requires. | ACTIVE | 0 / 2 |
| NG-4 | armdroid does not ship a hosted, networked, or multi-tenant control plane / web service; the delivered surface is the local `armdroid` CLI and library. | ACTIVE | 0 / 2 |

*Rationale:*
- **NG-1** — safety and liability: an unproven policy exploring on real hardware can
  drive joints past their limits and damage the arm or its surroundings. Learning stays
  in sim; the sim-to-real bridge (calibration + domain randomization) is the deliberate
  crossing point.
- **NG-2** — focus: the value is a tight, well-tested arm stack, not a middleware
  ecosystem. ROS/other transports are welcome as *driver* plug-ins, not as a rewrite of
  the core.
- **NG-3** — dependency risk and reproducibility: contributors and CI must be able to
  install and test the platform without a GPU or vendor SDK.
- **NG-4** — focus and attack surface: a networked control plane is a separate product
  with its own security and operational burden (Axes 7-8 in PHASES.md are Stretch).

## 4. Invariants (must not drift — enforce in review)

These formalize the "Non-negotiable conventions" already enforced in
[`AGENTS.md`](AGENTS.md). Carve-outs (§7) may expand scope; they may never relax an
invariant.

1. **Extensibility via registries** — new drivers, environments, planners, and RL agents
   are added through the subsystem `registry.py` + PEP 621 entry-point groups
   (`armdroid.drivers`, `armdroid.environments`, `armdroid.vec_environments`,
   `armdroid.planners`, `armdroid.rl_agents`). No `if / elif` type ladders in factory
   code; the orchestration composition root is not edited per addition.
2. **Compatibility window** — SemVer. `v0.2.x` is the v0.1 -> v0.2 migration window;
   breaking removals wait for the v0.4.0 track. A moved public symbol keeps its old path
   as a `DeprecationWarning`-emitting shim (`armdroid.compat`) for at least one minor
   release, pinned by `tests/regression/` and recorded in `CHANGELOG.md`.
3. **Testability via Protocols** — cross-subsystem APIs land in
   `armdroid.domain.protocols` as `@runtime_checkable Protocol` types before their
   implementation; unit tests use real fakes (`MockArmDriver`,
   `make_fake_isaac_env`, `RecordingTelemetry`) and require no hardware, GPU, or live
   services.
4. **Layering / domain purity** — the domain layer holds pure protocols, state types,
   and the error hierarchy with **no internal imports**; stateful and I/O concerns live
   in the adapter subsystems and orchestration, never in the domain.
5. **Configuration over magic numbers** — every operational value (numeric, string,
   device, dtype, capacity) is a Pydantic v2 field with documented bounds, a `Final`
   module constant for stable contracts, or a registry mapping. No literals in business
   code.
6. **Quality gates stay green** — `ruff check src/ tests/`, `mypy --strict src/`, and
   `pytest -m "not hardware and not isaac" --cov=armdroid --cov-fail-under=85` all pass.
   The only permitted exclusions are the documented ones (the pre-existing UP038 on
   `tests/helpers/fake_serial.py` and the Isaac-runtime mypy override block). Optional
   extras stay lazily imported so the default install never breaks.
7. **Security** — no secrets, credentials, or machine fingerprints are committed;
   operational credentials and endpoints live in runtime config, not the repo.

## 5. Long-term roadmap (themes, not dates)

> Detailed, changeable milestones live in [`NEXT_STEPS.md`](NEXT_STEPS.md) and the axis
> detail in [`docs/architecture/PHASES.md`](docs/architecture/PHASES.md); keep the three
> aligned — milestone detail changes there, not here.

- **M1 — Enterprise runtime foundation:** layered stack, stable public API, domain
  layer, registry seams, compatibility shims. *(landed, v0.2.x)*
- **M2 — Sim-to-real hardening:** interactive calibration, domain randomization, PnP
  6-DoF perception, Jetson edge deployment + health probe. *(landed, current branch)*
- **M3 — Backend enrichment:** additional drivers (dynamixel, ros2, dual-arm), tasks
  (pick-and-place, peg-in-hole), and planners (fast-downward, pddlstream) through the
  existing seams.
- **M4 — Scale & accelerated training:** vectorized Isaac training, distributed RL
  backends (Ray RLlib / JAX), broader telemetry (Prometheus / OTel traces).
- **M5 — Packaging & release:** v0.4.0 breaking-change track, deprecation-shim retirement,
  release automation.

## 6. How agents and contributors use this document

- Read this **before** planning a task; keep changes consistent with §3 scope and §4
  invariants.
- Put concrete, changeable to-dos in [`NEXT_STEPS.md`](NEXT_STEPS.md), not here.
- When a change would violate an invariant or expand scope, **surface it for a human
  decision** — do not implement first and ratify later.
- Agents must treat the §7 budgets as hard constraints: if a proposed task requires a
  carve-out and the relevant budget is exhausted or a trigger has fired, the correct
  output is a *charter-review request*, not an amendment draft.

## 7. Amendment Protocol (carve-out budget & review triggers)

Non-goals are amended only through **carve-outs**: dated, ratified, bounded exceptions
appended as blockquotes under §3. The default stance always remains the non-goal; the
carve-out is the exception.

### 7.1 Required carve-out format

Every carve-out MUST contain all of the following, or it is invalid and unenforceable:

> **Carve-out CO-{n} (deliberate amendment — ratified by {ROLE} on {DATE} per §7,
> against NG-{id}): {short title}.**
> {One paragraph: exactly what is now permitted, and for which component only.}
> Constraints (all required):
> - **{Bounding constraint}** — what the exception explicitly does NOT permit.
> - **{Mechanism constraint}** — how it is added (via the §4 extension seams, never by
>   editing core).
> - **{Safety/audit constraint}** — gates, confirmations, logging, default-off flags.
> - **{Sequencing constraint}** — what must land before this ships, if anything.
>
> This remains a bounded exception; every other §3 non-goal still stands.

### 7.2 Budgets (hard limits, not guidelines)

- **Per-non-goal budget:** at most **2 active carve-outs** per non-goal ID. A **third
  proposed amendment to the same non-goal automatically closes the amendment path** for
  that non-goal and forces a full charter review (§7.4). The review must choose one of:
  (a) rewrite the non-goal to honestly reflect the new scope, (b) reject the proposal
  and reaffirm the boundary, or (c) split the capability into a separate project with
  its own charter. "Ratify a third carve-out" is not an available outcome.
- **Global density trigger:** **3 or more ratified carve-outs within any rolling 90-day
  window — across all non-goals — forces a full charter review.** This catches the
  ratchet that per-ID counting misses: many small exceptions to *different* boundaries
  that together transform what the system is.
- **Cumulative trigger:** when total active carve-outs reach **50% of the number of
  non-goals** (currently 2 of 4), the next amendment proposal forces a review regardless
  of the other two triggers.

### 7.3 Anti-gaming rules

- **No splitting.** A proposal may not be divided into multiple smaller carve-outs to
  stay under budget; reviewers should count intent, not paperwork.
- **No non-goal inflation.** Adding new narrowly-worded non-goals solely to reset
  denominators for the cumulative trigger is itself a charter change requiring review.
- **Carve-outs expire on rewrite.** After a charter review rewrites a non-goal, its
  prior carve-outs are folded into the new text and retired; counters reset only
  through review, never administratively.

### 7.4 Full charter review (what a trigger forces)

A full charter review is a deliberate, human-led session (not an agent task) that:

1. Re-reads §1-§3 against what the system has actually become.
2. Rewrites Vision/Mission/Scope so the *default text* — not the exception list —
   describes reality.
3. Re-ratifies or retires each invariant.
4. Updates the Executive Summary and the Carve-out Ledger (§8).
5. Records the review date in §0. Until the review completes, **no new carve-outs may
   be ratified** and agents must treat pending amendment proposals as blocked.

### 7.5 Ratification authority

- Amendments are ratified by the **project maintainer**; charter reviews require the
  **maintainer plus one independent reviewer**. An amendment ratified by the same person
  who proposed it, on the same day, should be treated as a smell even when procedurally
  valid — the protocol works only if ratification is a genuine checkpoint, not a
  formality.

## 8. Carve-out Ledger (append-only)

| CO | Date | NG | Title | Ratified by | Status |
|----|------|----|-------|-------------|--------|
| — | — | — | *(none yet)* | — | — |

**Budget state:** 0 active carve-outs / 4 non-goals · last 90-day window: 0
ratifications · last full review: never.
