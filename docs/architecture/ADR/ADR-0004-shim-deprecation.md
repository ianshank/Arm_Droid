# ADR-0004: Shim deprecation strategy — backwards-compatible refactor through v0.4.0

- **Status**: Accepted
- **Date**: 2026-05-10
- **Deciders**: Maintainers
- **Supersedes**: n/a (new policy, Phase 3)
- **Superseded by**: n/a

## Context

Phase 1–2 moved types and factories to canonical paths (`armdroid.domain.*`,
`armdroid.orchestration.*`, `armdroid.hardware.esp32.*`, `armdroid.config.schema.*`). Any external
code that used the old paths (`armdroid.protocols`, `armdroid.factory`, `armdroid.orchestrator`,
`armdroid.hardware.esp32_json_driver`) would silently break on upgrade.

## Decision

Every moved module is replaced by a **shim** that:

1. Emits `DeprecationWarning` at import time with a `stacklevel=2` so the warning points at the
   caller's `import` statement (not into the shim itself).
2. Re-exports all previously public names so existing `from armdroid.protocols import X` calls
   still resolve.
3. Uses a `# ruff: noqa: E402` header to suppress the E402 lint rule (imports after non-import
   statements at module level).

Example shim pattern:

```python
# ruff: noqa: E402
import warnings

warnings.warn(
    "Importing from armdroid.protocols is deprecated and will be removed in v0.4.0. "
    "Import from armdroid.domain.protocols, armdroid.domain.state, "
    "or armdroid.domain.errors instead.",
    DeprecationWarning,
    stacklevel=2,
)

from armdroid.domain.errors import ArmCommandRejected as ArmCommandRejected  # noqa: F401
# ... remaining re-exports
```

In addition, `armdroid.compat.LegacyArmDriverAdapter` wraps any `ArmDriverProtocol` implementation
and maps the pre-v0.2 method names (`start`, `stop`, `send_joint_command`, `get_joint_states`) to
the current API.

### Removal timeline

| Version | Action |
|---|---|
| v0.2.x (current) | Shims active, emit `DeprecationWarning` |
| v0.3.0 | Shims upgraded to `FutureWarning`; LegacyArmDriverAdapter moved to compat-only package |
| v0.4.0 | Shims deleted; `armdroid.compat` package archived |

### Testing

- `tests/unit/test_shim_deprecations.py` — four tests verify that each shim module emits
  `DeprecationWarning` on import.
- `tests/unit/test_legacy_driver_adapter.py` — thirteen tests verify delegation and warning
  behaviour of `LegacyArmDriverAdapter`.
- `tests/regression/test_baseline.py::TestNamedConstantsRegression` — verifies shim module
  attributes are still reachable (filtered DeprecationWarning with `@pytest.mark.filterwarnings`).

## Consequences

**Positive**:
- External code written against v0.1.x continues to work through v0.3.x with only a deprecation
  warning, giving downstream consumers a two-minor-version migration window.
- `warnings.warn(..., stacklevel=2)` gives actionable messages: the warning points to the caller's
  import, making it trivial to grep-and-replace.
- The shim pattern is mechanical and reviewable — a new junior contributor can add a shim without
  understanding the full domain model.

**Negative**:
- Shims are technically dead code paths that accumulate maintenance debt. The hard deletion
  scheduled for v0.4.0 prevents indefinite accumulation.
- `stacklevel=2` is correct for direct imports but may give misleading line numbers when the
  import is inside a function body or a `__init__.py` that itself imports from a shim.
