# ADR-0003: Generic Registry pattern for extension points

- **Status**: Accepted
- **Date**: 2026-05-10
- **Deciders**: Maintainers
- **Supersedes**: n/a (new capability, Phase 2a)
- **Superseded by**: n/a

## Context

Pre-refactor, adding a new arm driver, planner, or environment required editing `factory.py` to
thread a new `elif` branch. There was no plugin mechanism — out-of-tree packages could not
contribute implementations without forking.

## Decision

Introduce `armdroid._registry.Registry[T]` — a generic, thread-safe registry mapping string keys
to factory callables for type `T`.

```python
class Registry(Generic[T]):
    def register(self, name: str, factory: Callable[..., T]) -> None: ...
    def create(self, name: str, *args: Any, **kwargs: Any) -> T: ...
    def available(self) -> list[str]: ...
```

Each subsystem owns a singleton registry populated at import time:

| Registry variable | Located in | Key used |
|---|---|---|
| `driver_registry` | `armdroid.hardware.registry` | `"mock"`, `"esp32"` |
| `planner_registry` | `armdroid.planning.registry` | `"pyperplan"` |
| `perception_registry` | `armdroid.perception.registry` | `"default"` |
| `environment_registry` | `armdroid.environments.registry` | `"tower_of_hanoi"`, `"laundry_sorting"` |
| `rl_agent_registry` | `armdroid.control.registry` | `"sac"` |

Entry-point groups mirror the registry keys so that `pip install my-arm-plugin` can contribute a
new driver without any change to armdroid's own source:

```toml
[project.entry-points."armdroid.drivers"]
my_arm = "my_arm_plugin.driver:MyArmDriver"
```

`orchestration.factory.build_arm_driver()` dispatches through `driver_registry.create(name, cfg)`
instead of a hardcoded `if/elif` chain.

## Consequences

**Positive**:
- New implementations are installed, not forked.
- `available_drivers()` / `available_planners()` etc. enumerate all registered keys for
  introspection and CLI `--list` subcommands (Phase 8).
- `Registry[T]` is unit-tested independently of any concrete implementation.

**Negative**:
- Entry-point loading (`importlib.metadata`) adds ~2 ms to first access on CPython 3.11. Acceptable
  for a CLI-dominated workload.
- Registry keys are strings — typos fail at runtime, not at import. Mitigated by `available()`
  validation in `factory.py` which raises `ArmConfigurationError` with a "did you mean?" hint.
