# ADR-0002: Domain layer — pure types, protocols, and error hierarchy

- **Status**: Accepted
- **Date**: 2026-05-10
- **Deciders**: Maintainers
- **Supersedes**: portion of ADR-0001 (elaborates the domain boundary)
- **Superseded by**: n/a

## Context

Phase 1 of the enterprise refactor extracted `src/armdroid/protocols.py` (≈450 LOC mixed types +
contracts) into a dedicated `armdroid.domain` sub-package. The goal was to create a stable,
import-free foundation layer that all other layers can depend on without triggering transitive
imports of heavy optional libraries (MuJoCo, torch, ultralytics, pyserial).

## Decision

`armdroid.domain` is a **pure-Python, no-internal-dependency** package. It comprises four modules:

| Module | Contents |
|---|---|
| `armdroid.domain.state` | `ArmState`, `PlanStep`, `SymbolicState`, `DetectedObject` |
| `armdroid.domain.errors` | `ArmError` hierarchy — `ArmDriverError`, `ArmCommandRejected`, `ArmPlanningError`, `ArmPerceptionError`, `ArmConfigurationError` |
| `armdroid.domain.protocols` | `@runtime_checkable Protocol` types: `ArmDriverProtocol`, `ArmControllerProtocol`, `ArmPlannerProtocol`, `ArmEnvironmentProtocol`, `ArmPerceptionProtocol` |
| `armdroid.domain.units` | Physical unit type aliases (`Radians`, `Metres`, `Seconds`) |

Rules enforced by ruff + CI:

- `armdroid.domain.*` must not import from any other `armdroid.*` module.
- All concrete implementations (`hardware`, `planning`, `perception`, `control`, `environments`)
  import **from** domain but are never imported **by** domain.
- Optional-dependency imports in concrete modules use `TYPE_CHECKING` guards or are deferred into
  function bodies.

## Consequences

**Positive**:
- `import armdroid.domain.state` resolves in < 10 ms on cold interpreter — no torch/MuJoCo init.
- Protocol types are testable with plain `object` stubs — no mock framework required.
- The error hierarchy's common `ArmError` base lets callers catch all armdroid errors with a
  single `except ArmError` clause.

**Negative**:
- Canonical import paths are longer (`armdroid.domain.protocols.ArmDriverProtocol` vs the old
  `armdroid.protocols.ArmDriverProtocol`). Mitigated by backwards-compatible shims (see ADR-0003).
- Maintaining the no-import rule requires discipline; currently enforced only by convention and CI
  review. Future: add ruff `banned-imports` rule.
