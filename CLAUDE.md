# CLAUDE.md - Claude Code project instructions for armdroid

This file is auto-loaded by Claude Code when working in this repo.
It supplements [`AGENTS.md`](AGENTS.md) with Claude-Code-specific
guidance. Read both before making changes.

---

## Repository compass

- **Working directory:** the project root contains `src/`, `tests/`,
  `firmware/`, `docs/`, `config/`, `scripts/`, `assets/`.
- **Source root:** `src/armdroid/` (PEP 420 namespace not used;
  `armdroid` is a regular package).
- **Test root:** `tests/` with sub-trees `unit/`, `property/`,
  `regression/`, `integration/`, `contract/`, `hardware/`, `isaac/`.
- **Python:** 3.11 minimum (Isaac Lab 2.3 pins 3.11). `pyproject.toml`
  declares 3.11 and 3.12 as the supported window.
- **Package manager:** plain `pip install -e ".[dev]"`. No Poetry / uv
  / pipenv adoption planned.

---

## Quality gates Claude Code MUST run before claiming a task is done

```bash
# Lint (zero errors expected on src/ and tests/)
python -m ruff check src/ tests/ --exclude tests/helpers/fake_serial.py

# Type check (strict mode, ~107 source files)
python -m mypy --strict src/

# Default test suite + coverage gate (85% line minimum, currently ~96%)
python -m pytest -m "not hardware and not isaac" --cov=armdroid --cov-fail-under=85 -q
```

If any of the three fails, fix it before declaring done. Do not bypass
hooks (`--no-verify`), do not skip mypy with `# type: ignore` unless
the suppressed error is documented inline.

---

## File / line / commit hygiene

- **Markdown links over bare references**: write
  `[file.py:123](src/armdroid/file.py:123)` so the user can click
  through.
- **Conventional commits**: `feat(<scope>): ...`, `fix(<scope>): ...`,
  `test(<scope>): ...`, `docs(<scope>): ...`, `chore(<scope>): ...`.
  Trailing `Co-Authored-By:` line is the norm.
- **One behavioural diff per commit**: peer reviewers expect to be
  able to revert one commit cleanly. Combine refactor + behaviour
  change into the same commit only when the refactor has no
  observable diff.
- **ASCII only** in docstrings + comments (RUF003 enforcement). Use
  straight quotes, ASCII hyphens, no curly quotes / EN DASH / EM DASH.

---

## Where to look first

| If the task involves... | Read... |
| --- | --- |
| Adding a backend (driver, env, planner, agent) | [`AGENTS.md`](AGENTS.md) "Common workflows" + the matching `registry.py` |
| Changing a public protocol | [`docs/architecture/ADR/`](docs/architecture/ADR/) - find the closest precedent, then write a new ADR |
| Adding a new axis | [`docs/architecture/PHASES.md`](docs/architecture/PHASES.md) "How to add a new axis" |
| Vec / Isaac work | [`ADR-0006`](docs/architecture/ADR/ADR-0006-vec-env-protocol.md), [`registry_vec.py`](src/armdroid/environments/registry_vec.py), [`reach_vec.py`](src/armdroid/environments/isaac/reach_vec.py) |
| Telemetry / observability | [`armdroid.telemetry`](src/armdroid/telemetry/__init__.py) - add SPAN constants there, never hardcoded |
| Config schemas | [`src/armdroid/config/schema/`](src/armdroid/config/schema) - Pydantic v2 with bounds on every numeric field |
| Physical Calibration & Limits | [`scripts/calibrate_arm.py`](scripts/calibrate_arm.py) - interactive joint extreme calibration script |
| Jetson Edge Deployment & Health | [`docs/deployment/JETSON.md`](docs/deployment/JETSON.md), [`Dockerfile`](Dockerfile), [`docker-compose.jetson.yml`](docker-compose.jetson.yml), [`scripts/jetson_health_check.py`](scripts/jetson_health_check.py) |

---

## Things to avoid

- **No hardcoded values.** If you type a literal magic number into
  business code, route it through Pydantic config instead. See
  [`AGENTS.md`](AGENTS.md) "Non-negotiable conventions" #1.
- **No reaching through private attributes.** F1 removed the
  `env._isaac_env` reach-through specifically. If you need an
  underlying handle, expose a documented accessor (see
  `VecArmEnvironmentProtocol.as_runner_env`).
- **No `if task_type == "..."` ladders.** Use the registry mapping
  pattern (e.g. `_VEC_TASK_REGISTRY_NAMES`).
- **No mocked-database / mocked-driver shortcuts in integration tests.**
  Real fakes (e.g. `MockArmDriver`, `make_fake_isaac_env`) exist for
  a reason; reach for them before reaching for `MagicMock`.
- **No `# type: ignore` without a reason.** If mypy strict complains,
  the right fix is almost always to tighten the type, not silence the
  check. Document any remaining ignore with a `# type: ignore[code] - reason`
  inline comment.

---

## Useful skills (when available)

- `superpowers:writing-plans` - draft an implementation plan before
  any non-trivial work.
- `superpowers:executing-plans` - execute a written plan task by task.
- `superpowers:subagent-driven-development` - dispatch fresh subagents
  per task for higher fidelity.
- `feature-dev:code-reviewer` - peer review the diff before pushing.
- `coderabbit:code-review` - hosted review pass (user-triggered).

---

The current branch `feat/production-blockers-sim2real` resolves all blockers for physical production (Sim-to-Real). It delivers interactive physical calibration (`calibrate_arm.py`), configurable domain randomization (friction, mass, gains) in the environments, OpenCV-based 6-DoF Perspective-n-Point orientation estimation in the perception PoseEstimator, and an end-to-end multi-stage L4T Docker deployment framework for NVIDIA Jetson devices with a dedicated health probe (`jetson_health_check.py`).

Before starting a new branch, check `NEXT_STEPS.md` for the active roadmap thread and `git log --oneline origin/main..HEAD` on any open PRs to avoid stepping on landing work.

