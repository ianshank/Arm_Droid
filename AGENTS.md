# AGENTS.md - armdroid contributor / AI-agent guide

This file is the canonical entry point for any agent (Claude Code,
Codex, Copilot CLI, etc.) working on the `armdroid` codebase. It
documents the conventions that automated changes MUST follow.

If you are a human contributor, the same conventions apply - the AI
agents are codifying what the maintainers already do by hand.

---

## Architecture in one screen

armdroid is a layered Python robot-arm platform:

| Layer | Path | Responsibility |
| --- | --- | --- |
| **Public API + CLI** | `armdroid`, `armdroid.api`, `src/armdroid/main.py` | SemVer-stable surface, argparse entry, top-level composition |
| **Orchestration** | `src/armdroid/orchestration/` | Composition root (`factory.py`), lifecycle (`orchestrator.py`), driver-kind resolution |
| **Adapter subsystems** | `src/armdroid/{hardware,planning,perception,control,environments}/` | Concrete implementations + per-subsystem registries |
| **Configuration** | `src/armdroid/config/` | Pydantic v2 `ArmSettings`, YAML overlay loader, asset paths |
| **Domain** | `src/armdroid/domain/` | Pure `@runtime_checkable` Protocols, state types, error hierarchy. NO internal imports. |
| **Compatibility** | `src/armdroid/compat/`, shim modules | Preserves legacy imports through the migration window |

All subsystem boundaries are `@runtime_checkable` Protocols. The
composition root in [`armdroid.orchestration.factory`](src/armdroid/orchestration/factory.py)
is the single wiring point. Plugins extend the system via PEP 621
entry-point groups (`armdroid.drivers`, `armdroid.environments`,
`armdroid.vec_environments`, `armdroid.planners`, `armdroid.rl_agents`).

See [`docs/architecture/C4.md`](docs/architecture/C4.md) for the C4
diagrams and [`docs/architecture/ADR/`](docs/architecture/ADR/) for the
decision log (ADR-0001 through ADR-0006).

---

## Non-negotiable conventions

These rules are enforced by review (and increasingly by tests). Break
them and the PR will be sent back.

### 1. No hardcoded values

Every numeric, string, device, dtype, or capacity that influences
runtime behaviour must come from:

- A Pydantic field on `ArmSettings` (or one of its sub-models) with
  documented bounds (`ge`, `le`, `gt`, `min_length`, `Literal[...]`).
- A module-level constant with a `Final` annotation when the value is
  a stable contract (e.g. SPAN names, registry kinds).
- A `frozenset` / `dict[str, str]` mapping when extension is expected
  (e.g. [`_VEC_TASK_REGISTRY_NAMES`](src/armdroid/orchestration/factory.py)).

If you find yourself typing a literal value into business code, route
it through one of the three above instead.

### 2. Protocol-first design

Cross-subsystem APIs land in `armdroid.domain.protocols` as
`@runtime_checkable Protocol` types **before** the implementation.
Concrete classes satisfy protocols structurally; explicit subclassing
is unusual.

When you add a new axis (Multi-arm, Multi-sim, Multi-planner, etc.),
follow the steps in
[`docs/architecture/PHASES.md#how-to-add-a-new-axis`](docs/architecture/PHASES.md):

1. Define the contract in `armdroid.domain.protocols`.
2. Add a `Registry[T]` + entry-point group in a sibling module.
3. Implement built-in backends; register on import.
4. Add the config discriminator (`cfg.<axis>.kind`).
5. Document under `docs/architecture/add-a-<kind>.md` (the project
   keeps extension guides next to the ADRs rather than in a separate
   `docs/extension/` tree).

### 3. Registry over hardcoded branches

Adding a new driver, env, planner, or agent must NOT require an `if /
elif` ladder in factory code. Register the factory in the
subsystem's registry, then resolve by name. The orchestration factory
threads typed configs in; the registry handles dispatch.

Exception: when a backend needs dual-config threading (e.g.
`isaac_sim` driver requiring both `ArmConfig` and `ArmSimIsaacConfig`),
the explicit factory branch is acceptable - see
`build_arm_driver` for the documented pattern.

### 4. Backwards-compatibility window

`v0.2.x` is the v0.1 -> v0.2 migration window. Behaviour-preserving
refactors land here; breaking removals wait for v0.4.0. When a public
symbol moves, the old path stays as a `DeprecationWarning`-emitting
shim (see `armdroid.compat`) for at least one minor release. Tests in
`tests/regression/` pin the contract.

### 5. Lazy optional imports

Optional extras (`[isaac]`, `[ble]`, `[telemetry]`) MUST stay opt-in.
A default `pip install -e ".[dev]"` must NOT fail because `isaaclab`
is missing. The pattern: lazy `import` inside the function body,
wrapped in `try / except ImportError -> ArmDriverError` with an
actionable install hint. See `_so_arm_reach_isaac_factory` and
`_so_arm_reach_isaac_vec_factory` for canonical examples.

### 6. Telemetry spans use module-level constants

Never hardcode span names at call sites. Add a `SPAN_*` constant in
`armdroid.telemetry.__init__` and export via `__all__`. Then:

```python
from armdroid.telemetry import SPAN_DRIVER_CONNECT, get_telemetry

with get_telemetry().start_span(SPAN_DRIVER_CONNECT, port=path):
    ...
```

---

## Quality gates

Every PR must pass these before merge. `make check` runs all three.

| Gate | Command | Threshold |
| --- | --- | --- |
| Lint | `ruff check src/ tests/` | Zero errors. Ruff 0.15.12, line-length 100, `select = ["E","W","F","I","N","UP","B","A","C4","SIM","RUF","PT","S","ANN","D","T20"]`. Pre-existing UP038 on `tests/helpers/fake_serial.py` is exempt. |
| Types | `mypy --strict src/` | Zero errors. Strict mode across all source files. Isaac-runtime modules have an explicit override block in `pyproject.toml` so the optional `isaaclab` types do not break strict checking. |
| Tests + coverage | `pytest -m "not hardware and not isaac" --cov=armdroid --cov-fail-under=85` | >= 85% line coverage. Currently ~96%. Markers `hardware`, `gpu`, `isaac` are excluded by default. |

Property tests (Hypothesis) live under `tests/property/`. Regression
tests under `tests/regression/` pin the contract. Isaac smoke tests
under `tests/isaac/` are auto-marked `isaac` + `gpu` by
`tests/isaac/conftest.py` and only run with `ARMDROID_ISAAC_RUN=1` +
the `[isaac]` extra installed.

### Lint trap: RUF003

Avoid Unicode lookalikes (EN DASH `U+2013`, EM DASH `U+2014`, curly
quotes) in docstrings + comments. They trigger RUF003 and have already
caused one CI failure (commit `7b1693d`). Use ASCII `-` and straight
quotes only.

---

## Test discipline

- **TDD by default**: write the failing test first, run it, implement
  the minimum to pass. Most F1 commits show this pattern explicitly.
- **No mock leaks**: tests under `tests/unit/` MUST NOT import
  `isaaclab`. Use `tests/helpers/fake_isaac_env.make_fake_isaac_env`
  via `monkeypatch.setattr(reach_vec, "_build_isaac_env", ...)`.
- **Markers**: `slow`, `hardware`, `gpu`, `isaac`, `regression`,
  `e2e`, `contract`. Mark new tests so the default `make test` stays
  fast.
- **Telemetry assertions**: reuse
  [`tests/helpers/recording_telemetry.RecordingTelemetry`](tests/helpers/recording_telemetry.py)
  rather than rolling your own provider stub.
- **Property invariants**: when adding a new env protocol, lift shared
  invariants into a sibling helper (e.g.
  [`tests/property/_vec_invariants.py`](tests/property/_vec_invariants.py))
  rather than duplicating across files.

---

## Common workflows for agents

### Add a new driver

1. Implement under `src/armdroid/hardware/<name>/driver.py` satisfying
   `ArmDriverProtocol`.
2. Register in `src/armdroid/hardware/registry.py` as a lazy factory.
3. Widen the `arm_driver_kind` Literal in `armdroid.config.schema.arm`
   (if user-facing).
4. Add entry point under `[project.entry-points."armdroid.drivers"]`
   in `pyproject.toml` if you want out-of-tree consumers to discover it.
5. Unit tests + contract tests (use
   `tests/contract/test_arm_driver_protocol_contract.py` if it exists,
   or mirror its pattern).

### Add a new environment

Same shape as driver. For a vectorised variant: see F1's
[`registry_vec.py`](src/armdroid/environments/registry_vec.py) and
[`reach_vec.py`](src/armdroid/environments/isaac/reach_vec.py). Update
[`_VEC_TASK_REGISTRY_NAMES`](src/armdroid/orchestration/factory.py).

### Add a new RL agent

Implement `ArmRLAgentProtocol`. If the agent needs vec training, also
implement `VecArmRLAgentProtocol` and add its algorithm key to
[`_VEC_CAPABLE_ALGORITHMS`](src/armdroid/orchestration/factory.py).

### Change a public protocol

This is a v0.4.0-track change. Open an ADR under
`docs/architecture/ADR/` describing the migration. Add deprecation
shims in `armdroid.compat`. Extend `tests/regression/` to pin both old
and new surfaces.

---

## When in doubt

- Read the most recent ADR (currently [ADR-0006](docs/architecture/ADR/ADR-0006-vec-env-protocol.md))
  for the latest worked example.
- Read [`docs/architecture/PHASES.md`](docs/architecture/PHASES.md)
  for the long-range axis roadmap.
- Read [`NEXT_STEPS.md`](NEXT_STEPS.md) for what is currently in
  flight.
- Look at the commit message convention (`feat(<scope>):`,
  `fix(<scope>):`, `test(<scope>):`, `docs(<scope>):`,
  `chore(<scope>):`) in `git log --oneline origin/main..HEAD`.

The repo's culture is "small, sharp, well-tested commits with one
behavioural diff per commit." Stick with that and reviewers will move
quickly.
