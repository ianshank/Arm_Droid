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

## Axis 4 — Multi-sim (Foundation in PR-A; Isaac integration lands in PR-B)

Current: MuJoCo via gymnasium env wrappers + **Isaac Sim 5.1 / Isaac Lab 2.3
scaffolding shipped in PR-A**: vendored SO-ARM101 simulation assets
(URDF + MJCF + STL meshes under `assets/so_arm/so101/`), the
`armdroid.config.paths` resolver, and the `arm_driver_kind` discriminator
field that PR-B widens to include `"isaac_sim"` alongside the actual
`IsaacSimDriver` registration.

Foundation seam: env registry + driver registry both serve as the
multi-sim seam. The `arm_driver_kind` discriminator decouples sim
selection from the legacy `mock_hardware` bool, and `ArmRLAgentProtocol`
gives non-SAC agents (RSL-RL PPO is next) a uniform contract.

Next: PR-B lands `IsaacSimDriver`, `SoArmReachIsaacEnv`,
`RslRlPpoAgent`, and the `[isaac]` extra. PyBullet / Drake remain as
future env-registry entries; their seam is the same.

## Axis 5 — Distributed / accelerated training (Planned)

Current: SB3 SAC single-process.

Foundation seam: `armdroid.control.rl.registry` + entry-point group
`armdroid.rl_agents` so PPO/TD3/custom agents register the same way.

Future: Ray RLlib backend, DeepMind Acme adapter, JAX (Brax/Mctx) agents,
multi-GPU vectorized envs.

## Axis 6 — Telemetry & observability (Foundation)

Current: structlog JSON logging + **telemetry seam implemented** (Phase 0.3).

Foundation seam: `armdroid.telemetry` package — no-op by default (`NullTelemetry`
with `contextlib.nullcontext`); `OtelTelemetry` backend available under
`armdroid[telemetry]` (`opentelemetry-api/sdk>=1.27`).
`Esp32JsonDriver.connect`, `disconnect`, and `send_joint_positions` are all
instrumented.  See [`src/armdroid/telemetry/`](../../src/armdroid/telemetry/).

Future: Prometheus metrics endpoint, OTel traces (planner solve, env step),
structured event sinks (Kafka / Loki).

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

See [ADR-0003: Generic Registry pattern for extension points](ADR/ADR-0003-registry-pattern.md)
for the canonical pattern.
