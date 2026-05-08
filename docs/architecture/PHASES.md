# Future Phases

This document enumerates the future axes of growth for armdroid. The
[enterprise refactor](ADR/ADR-0001-enterprise-layering.md) creates extension
seams (registries + entry-points) for each axis so subsequent phases land as
plug-ins rather than core rewrites.

> Status legend: **Foundation** = seam exists in v0.2.0 (this refactor),
> **Planned** = roadmap, **Stretch** = exploratory.

## Axis 1 — Multi-arm hardware (Foundation)

Current: single-arm protocol (`ArmDriver`) with `MockArmDriver` and
`Esp32JsonDriver` implementations.

Foundation seam: `armdroid.hardware.registry` + entry-point group
`armdroid.drivers`. Selected via `cfg.arm.driver.kind`.

Future drivers (Planned):
- `dynamixel` — U2D2 / OpenCM bridge for Robotis servo arms.
- `ros2` — ROS 2 control bridge.
- `mujoco_sim` — direct sim driver (skip serial, drive MuJoCo joints).
- `dual_arm` — composite driver wrapping two `ArmDriver` instances.

## Axis 2 — Multi-environment (Foundation)

Current: `tower_of_hanoi`, `laundry_sorting`.

Foundation seam: `armdroid.environments.registry` + entry-point group
`armdroid.environments`. Wrappers (`curriculum`, `domain_randomizer`,
`reward_shaping`) compose orthogonally.

Future tasks (Planned): pick-and-place, peg-in-hole, bin-picking, cable
routing, deformable manipulation.

## Axis 3 — Multi-planner (Foundation)

Current: pyperplan symbolic planner + LLM replanner stub (Anthropic).

Foundation seam: `armdroid.planning.registry` + entry-point group
`armdroid.planners` and `armdroid.planning.backends.llm.base.ReplannerBackend`
protocol.

Future backends (Planned): `fast-downward`, `pddlstream`, OpenAI replanner,
local LLM replanner, hybrid task-and-motion planner.

## Axis 4 — Multi-sim (Stretch)

Current: MuJoCo via gymnasium env wrappers.

Foundation seam: env registry doubles as sim seam (sim is just a different
env source).

Future: Isaac Sim, PyBullet, Drake — each surfaced as an env-registry entry.

## Axis 5 — Distributed / accelerated training (Planned)

Current: SB3 SAC single-process.

Foundation seam: `armdroid.control.rl.registry` + entry-point group
`armdroid.rl_agents` so PPO/TD3/custom agents register the same way.

Future: Ray RLlib backend, DeepMind Acme adapter, JAX (Brax/Mctx) agents,
multi-GPU vectorized envs.

## Axis 6 — Telemetry & observability (Foundation)

Current: structlog JSON logging.

Foundation seam: `armdroid.telemetry` package — no-op by default; OTel
provider wires up under `armdroid[telemetry]` extras.

Future: Prometheus metrics endpoint, OTel traces (driver send, planner
solve, env step), structured event sinks (Kafka / Loki).

## Axis 7 — Control plane / API surface (Stretch)

Current: CLI only (`armdroid` entry point).

Future: gRPC service exposing `ArmDriver` protocol over the wire (treat
remote arm as just another driver via `grpc` driver kind), REST/WebSocket
gateway for web UI, multi-tenant orchestrator.

## Axis 8 — Web UI (Stretch)

Future: dashboard for live joint state, plan execution, episode replay,
training curve visualisation. Driven by control-plane API above.

## How to add a new axis

1. Define the contract in `src/armdroid/domain/protocols.py` (or a sub-namespace).
2. Add a registry module + entry-point group.
3. Implement built-in backends; register on import.
4. Add config discriminator (`cfg.<axis>.kind`).
5. Document under `docs/extension/add-a-<kind>.md`.

See [ADR-0002 Plugin registries](ADR/ADR-0002-plugin-registries.md) (forthcoming
in Phase 2) for the canonical pattern.
