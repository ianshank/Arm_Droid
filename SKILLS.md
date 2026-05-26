# SKILLS.md - armdroid project skill registry

This file documents the **project-scoped capabilities** (skills) that
agents and contributors should reach for when working on armdroid. It
is the project counterpart to [`AGENTS.md`](AGENTS.md) and
[`CLAUDE.md`](CLAUDE.md):

- `AGENTS.md` - architectural conventions and quality gates.
- `CLAUDE.md` - Claude-Code-specific instructions.
- `SKILLS.md` (this file) - reusable capabilities the project relies on.

If a "skill" is not listed here it's either (a) not relevant to this
codebase, or (b) something we have not yet adopted. Adding a new skill
means adding it here AND wiring it into the matching workflow.

---

## In-repo skills

These live in the codebase and can be invoked by name.

### Architecture skills

| Skill | Where | When to use |
| --- | --- | --- |
| **Registry-pattern extension** | [`armdroid._registry.Registry`](src/armdroid/_registry.py) + per-subsystem `registry.py` files | Adding a new backend (driver, env, planner, agent, vec env). |
| **Protocol-first design** | [`armdroid.domain.protocols`](src/armdroid/domain/protocols.py) | Defining or changing a cross-subsystem boundary. |
| **Lazy optional-import gate** | [`_so_arm_reach_isaac_factory`](src/armdroid/environments/registry.py), [`_so_arm_reach_isaac_vec_factory`](src/armdroid/environments/registry_vec.py) | Adding any code that imports `isaaclab`, `bleak`, `opentelemetry`, etc. |
| **AppLauncher singleton coordination** | [`armdroid.hardware.isaac_sim._app_state`](src/armdroid/hardware/isaac_sim/_app_state.py), [`SoArmReachIsaacVecEnv._ensure_built`](src/armdroid/environments/isaac/reach_vec.py) | Building a new Isaac runtime entry point (driver, env, agent). |
| **OpenCV PnP 6-DoF estimation** | [`armdroid.perception.pose_estimator`](src/armdroid/perception/pose_estimator.py) | Upgrading PoseEstimator from a zero placeholder to full 3D translation/rotation. |
| **Physical domain randomization** | [`armdroid.environments.isaac._domain_randomization`](src/armdroid/environments/isaac/_domain_randomization.py) | Randomizing link masses, frictions, and joint controller PD gains. |


### Telemetry skills

| Skill | Where | When to use |
| --- | --- | --- |
| **SPAN_\* constants pattern** | [`armdroid.telemetry.__init__`](src/armdroid/telemetry/__init__.py) | Adding a new instrumented operation. Never hardcode span names. |
| **OtelTelemetry backend** | [`armdroid.telemetry.otel`](src/armdroid/telemetry/otel.py) | Wiring up live tracing under the `[telemetry]` extra. |

### Testing skills

| Skill | Where | When to use |
| --- | --- | --- |
| **`RecordingTelemetry` provider stub** | [`tests/helpers/recording_telemetry.py`](tests/helpers/recording_telemetry.py) | Any test that asserts on span emission. |
| **`make_fake_isaac_env` fixture** | [`tests/helpers/fake_isaac_env.py`](tests/helpers/fake_isaac_env.py) | Any test that needs a `ManagerBasedRLEnv`-shaped mock without importing `isaaclab`. |
| **Hypothesis property invariants** | [`tests/property/_vec_invariants.py`](tests/property/_vec_invariants.py), [`tests/property/test_arm_env_invariants.py`](tests/property/test_arm_env_invariants.py) | New env or driver - lift shape / invariant assertions into a sibling helper rather than duplicating. |
| **`mock_settings` / `_mock_arm_driver_env`** | [`tests/conftest.py`](tests/conftest.py) | Autouse fixtures that force the mock driver path in unit tests. |
| **Isaac-gated smoke pattern** | [`tests/isaac/conftest.py`](tests/isaac/conftest.py) | Tests that require a real GPU + `[isaac]` extra. Gated by `ARMDROID_ISAAC_RUN=1`. |
| **HIL hardware tests** | [`tests/hardware/conftest.py`](tests/hardware/conftest.py) | Real ESP32 hardware-in-the-loop tests. Gated by `ARMDROID_HIL_RUN=1`. |

### Tooling skills

| Skill | Where | When to use |
| --- | --- | --- |
| **`make check`** | [`Makefile`](Makefile) | The canonical pre-PR gate: lint + types + coverage. |
| **`scripts/check_isaac_install.py`** | [`scripts/check_isaac_install.py`](scripts/check_isaac_install.py) | Pre-install probe for the `[isaac]` extra - safe on any host. |
| **`scripts/gen_firmware_config.py`** | [`scripts/gen_firmware_config.py`](scripts/gen_firmware_config.py) | Codegen for the ESP32 firmware config header. |
| **HIL runner scripts** | [`scripts/run_hil.sh`](scripts/run_hil.sh), [`scripts/run_hil.ps1`](scripts/run_hil.ps1) | One-command hardware gate. |
| **`scripts/calibrate_arm.py`** | [`scripts/calibrate_arm.py`](scripts/calibrate_arm.py) | Interactive extreme joint range and homing calibration utility. |
| **`scripts/jetson_health_check.py`** | [`scripts/jetson_health_check.py`](scripts/jetson_health_check.py) | Self-diagnostic CLI probe for Jetson Docker healthchecks (CUDA, UART, RealSense). |


---

## Agent skills (when available in the environment)

These are agent-side capabilities, not in-repo files. They depend on
which Claude Code plugins / skills are installed locally. If a skill
listed here is not available, fall back to the in-repo equivalent
above.

| Skill | Use for |
| --- | --- |
| `superpowers:brainstorming` | Exploring requirements before any non-trivial change. |
| `superpowers:writing-plans` | Drafting a concrete implementation plan (saved under `docs/superpowers/plans/`). |
| `superpowers:executing-plans` | Executing a written plan task-by-task with checkpoints. |
| `superpowers:subagent-driven-development` | Same, but with a fresh subagent per task for higher fidelity. |
| `superpowers:verification-before-completion` | Re-asserting the verification commands before claiming done. |
| `superpowers:systematic-debugging` | Use when a test fails for an unclear reason. |
| `feature-dev:code-reviewer` | Local peer review pass against the diff. |
| `coderabbit:code-review` | Hosted CodeRabbit review pass (user-triggered, billed). |
| `git-workflow:feature` / `release` / `hotfix` | Git Flow branch creation with proper naming. |

When writing or modifying skill metadata, see
[`docs/architecture/ADR/`](docs/architecture/ADR/) for the conventions
that govern public-API stability.

---

## How to add a new skill

1. **In-repo skill** (most cases):
   - Add the module under `src/armdroid/<subsystem>/` or
     `tests/helpers/`.
   - Document it in the matching subsystem README or directly in this
     file.
   - Add a regression test under `tests/regression/` if the skill is
     part of a public API surface.

2. **Agent-side skill** (rare; usually external):
   - Reference it in this file under "Agent skills" with a one-line
     "use for" description.
   - Update [`CLAUDE.md`](CLAUDE.md) if the workflow changes.

Do not register skills via global config files; this file is the
single source of truth for armdroid's skill graph.
