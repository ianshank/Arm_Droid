# ADR-0001: Enterprise layering & public API surface

- **Status**: Accepted
- **Date**: 2026-05-08
- **Deciders**: Maintainers (autonomous direction from project owner: "the whole enchilada")
- **Supersedes**: n/a (first ADR)
- **Superseded by**: n/a

## Context

armdroid v0.1.0 (post PR #1, ESP32-JSON arm transport) is a flat package:

```
src/armdroid/
  __init__.py        # __version__ only
  protocols.py       # 450 LOC — types + contracts mixed together
  factory.py         # build_*; imports concrete drivers directly
  orchestrator.py    # ArmOrchestrator
  main.py / __main__.py
  config/schema.py   # ~800 LOC god-module with mypy waivers
  control/   environments/   hardware/   logging/   perception/   planning/
  hardware/esp32_json_driver.py  # ~450 LOC god-module
```

This shape worked for one hardware target, one env, one planner, and one
trainer. It does not scale to the [future axes](../PHASES.md): multi-arm,
multi-env, multi-planner, multi-sim, distributed training, telemetry,
control-plane API. Specific pain points:

1. **No plugin seam.** Adding a new arm driver, planner, or env backend
   requires editing `factory.py` and threading a new branch.
2. **No public/internal split.** Every module is reachable; refactors break
   external code silently.
3. **Two god modules** (`config/schema.py`, `hardware/esp32_json_driver.py`)
   carry mypy waivers and resist further growth.
4. **Dual code paths** in primitives & drivers (`if dof < 7`) ship every
   commit, doubling the test surface and breeding subtle drift.
5. **Coverage floor of 85%** with no per-module gate lets individual
   subsystems decay quietly.

## Decision

Adopt a layered package layout with an explicit public façade and registry-based
plugin seams. Specifically:

1. **Layers** (in dependency order, top depends on bottom):
   - `armdroid.api` — stable public façade. Only path external users import.
   - `armdroid.cli` — command-line surface.
   - `armdroid.orchestration` — composition root (`factory`, `orchestrator`,
     `lifecycle`).
   - `armdroid.{hardware,planning,perception,control,environments}` — adapters
     and domain-specific logic. Each has a `registry.py` and a `backends/`
     sub-package.
   - `armdroid.config` — settings root + YAML loader. Schema split into
     per-subsystem files.
   - `armdroid.domain` — pure types and `Protocol` contracts. No I/O, no
     framework imports.
   - `armdroid.telemetry` — observability hooks (no-op default).
   - `armdroid.compat` — backwards-compatibility shims.

2. **Public API**: `from armdroid import …` resolves to `armdroid.api`. The
   façade re-exports a curated set: `ArmOrchestrator`, `ArmSettings`,
   `build_orchestrator`, `ArmDriver` protocol, error hierarchy, and
   `__version__`. Everything else is internal and may move between minor
   versions.

3. **Registries + entry points**: each adapter axis exposes
   `register_<kind>(name, factory)` / `get_<kind>(name, cfg)`. Built-in
   backends register on import; out-of-tree packages register via entry-point
   groups (`armdroid.drivers`, `armdroid.planners`, etc.).

4. **Compat layer (sunset window)**: legacy import paths
   (`armdroid.protocols`, `armdroid.factory`, `armdroid.orchestrator`) become
   thin re-export shims under `armdroid.compat` that emit `DeprecationWarning`
   on import. Sunset target: v0.4.0 (two minor releases after v0.2.0).

5. **Coverage floor**: bump to 90% global with an additional 80% per-module
   gate enforced by a CI script.

## Consequences

### Positive

- New drivers / planners / envs land as plugins, not core edits.
- Public API surface is reviewable and SemVer-able.
- God modules decompose along documented seams; mypy waivers can be removed.
- Single code path per driver removes `if dof < 7` drift; legacy contract
  preserved via `compat.LegacyArmDriverAdapter`.
- 90/80 coverage floor catches per-subsystem decay early.

### Negative

- Larger initial diff. Mitigated by phased rollout (see
  [PHASES.md](../PHASES.md)) and re-export shims keeping all 429 existing
  tests green during Phase 1.
- Out-of-tree code that imports `armdroid.<internal>` paths breaks at v0.4.0.
  Mitigated by `DeprecationWarning` from v0.2.0 onward and a migration
  cookbook (`docs/migration/v0.1-to-v0.2.md`).
- Slight import-time cost for entry-point discovery (one-shot, ~tens of ms).
  Acceptable for a non-realtime composition root.

### Neutral

- Hatchling already supports entry-points in `pyproject.toml`; no build
  system change.
- structlog logging is unchanged; new layers gain a
  `logger = structlog.get_logger().bind(component=__name__)` convention.

## Alternatives considered

1. **Keep flat layout, add registries only**. Rejected: god modules
   continue to grow; no public/internal split.
2. **Split into multiple distributions** (`armdroid-core`,
   `armdroid-esp32`, …). Rejected for v0.2.0 — premature; entry-point seams
   give 90% of the benefit without release coordination overhead. Re-evaluate
   once first out-of-tree backend ships.
3. **Adopt Hydra for config**. Rejected: pydantic v2 + structlog already
   meet needs; Hydra would force every consumer to learn its overlay model.
4. **Drop legacy 6-DoF adapter outright (no compat)**. Rejected: PR #1 just
   shipped the dual path; consumers haven't had a deprecation window.

## References

- [PHASES.md](../PHASES.md) — future axes this layout enables.
- PR #1 — `feature/esp32-arm-integration` (merged 2026-05-08).
- pydantic v2 docs on `BaseSettings` composition.
- Python packaging: [PEP 621][pep621] entry-point groups.

[pep621]: https://peps.python.org/pep-0621/
