"""IsaacSimDriver — Isaac Lab 2.3 articulation as ArmDriverProtocol (PR-B B.9).

Lazy-imports ``isaaclab`` and ``torch`` inside ``connect()`` so the
module is importable on default installs (the ``[isaac]`` extra is
required only at runtime, not at import time). Mirrors the
``Esp32JsonDriver`` pattern for telemetry spans and structlog events.

Uses Isaac Lab 2.3's **singular** ``set_joint_position_target`` API
per <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html>.
The plural form is on the legacy ``omni.isaac.core.articulations.Articulation``
which we do not use. ADR-0005 records the verification.

AppLauncher process-singleton: Isaac Sim's Kit cannot be launched twice
in the same process. The shared ``_app_state`` module enforces this
with a clear ``ArmDriverError`` rather than letting the second call
produce inscrutable Kit double-init errors. The state is shared across
the driver, env, and any future Isaac-runtime entry points so an
env-first init does not cause a driver-second double-launch crash.
(Peer-review N-3 + PR-11 review fix C2.)

Coverage-omit: this module is in ``[tool.coverage.run].omit``. Tests
live under ``tests/isaac/`` and only run with ``ARMDROID_ISAAC_RUN=1``
+ a CUDA GPU. Contract tests under ``tests/contract/`` auto-skip the
``isaac_sim`` driver factory when ``isaaclab`` is missing or
``ARMDROID_ISAAC_RUN`` is unset.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from armdroid.domain.errors import ArmCommandRejected, ArmDriverError
from armdroid.domain.state import ArmState
from armdroid.hardware.isaac_sim import _app_state
from armdroid.hardware.isaac_sim.gripper import (
    normalised_vector_to_radians,
    radians_vector_to_normalised,
)
from armdroid.logging.setup import get_logger
from armdroid.telemetry import (
    SPAN_DRIVER_CONNECT,
    SPAN_DRIVER_DISCONNECT,
    SPAN_DRIVER_SEND,
    get_telemetry,
)

if TYPE_CHECKING:
    from armdroid.config.schema.arm import ArmConfig
    from armdroid.config.schema.sim_isaac import ArmSimIsaacConfig

_log = get_logger(__name__)


def _reset_app_launched_for_tests() -> None:
    """Reset the once-per-process AppLauncher guard. Tests only.

    Kept as a thin shim over :func:`_app_state.reset_for_tests` for
    backwards compatibility with existing test fixtures.
    """
    _app_state.reset_for_tests()


class IsaacSimDriver:
    """ArmDriverProtocol-compliant driver backed by Isaac Lab 2.3 / Isaac Sim 5.1.

    Construction is cheap and does NOT load ``isaaclab`` — the heavy
    imports happen inside :meth:`connect`. This means
    ``IsaacSimDriver(cfg.arm)`` is safe to call from a default-install
    smoke test; the exception only fires when the driver is actually
    used (``await drv.connect()``).

    Args:
        cfg: Arm hardware configuration. ``cfg.dof`` and
            ``cfg.joint_limits`` are honoured.
        sim_cfg: Optional Isaac Sim configuration. Defaults to
            :func:`armdroid.config.schema.sim_isaac._default_sim_cfg`.
    """

    def __init__(
        self,
        cfg: ArmConfig,
        *,
        sim_cfg: ArmSimIsaacConfig | None = None,
    ) -> None:
        # Lazy default — imports kept local so this construction stays
        # cheap on default installs without [isaac].
        if sim_cfg is None:
            from armdroid.config.schema.sim_isaac import _default_sim_cfg

            sim_cfg = _default_sim_cfg()

        # ArmDriverProtocol assumes a single arm — vectorised training
        # (num_envs > 1) bypasses the driver and uses Isaac Lab's runner
        # directly. Reject the misconfiguration up front so the failure is
        # actionable; otherwise send_joint_positions() reshape((num_envs,
        # -1)) would raise an opaque torch error on the first command.
        if sim_cfg.num_envs != 1:
            msg = (
                f"IsaacSimDriver requires sim_cfg.num_envs == 1 (protocol "
                f"path is single-arm); got num_envs={sim_cfg.num_envs}. "
                "For vectorised training, bypass the driver/protocol and "
                "use Isaac Lab's OnPolicyRunner against the raw env."
            )
            raise ArmDriverError(msg)

        self._cfg = cfg
        self._sim_cfg = sim_cfg
        self._dof = cfg.dof
        self._articulation: Any = None
        self._sim_context: Any = None
        self._app_launcher: Any = None
        self._connected = False
        self._estop_latched = False
        _log.info(
            "isaac_sim_driver_init",
            dof=self._dof,
            headless=self._sim_cfg.headless,
            num_envs=self._sim_cfg.num_envs,
            urdf_fallback_path=str(self._sim_cfg.urdf_fallback_path),
        )

    @property
    def is_connected(self) -> bool:
        """Whether the simulator is initialised and ready to receive commands."""
        return self._connected

    @property
    def dof(self) -> int:
        """Degrees of freedom (joint vector length)."""
        return self._dof

    # ------------------------------------------------------------------ #
    # Modern lifecycle
    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        """Boot Kit + load the SO-ARM articulation. Idempotent."""
        if self._connected:
            return
        with get_telemetry().start_span(
            SPAN_DRIVER_CONNECT,
            urdf_fallback_path=str(self._sim_cfg.urdf_fallback_path),
            headless=self._sim_cfg.headless,
        ):
            _log.info(
                "isaac_sim_driver_connecting",
                urdf_fallback_path=str(self._sim_cfg.urdf_fallback_path),
            )
            if _app_state.is_app_launched():
                msg = (
                    "Isaac Sim Kit already initialised in this process "
                    "(by an earlier driver, env, or out-of-band AppLauncher); "
                    "Kit cannot be launched twice per process."
                )
                _log.error("isaac_sim_driver_app_launcher_failed", error=msg)
                raise ArmDriverError(msg)
            try:
                # Probe import — fails fast with a clear message when
                # the [isaac] extra is missing, before we get to the
                # heavier _launch_kit_and_articulation thread call.
                import isaaclab.app  # noqa: F401  (probe only; real import in _launch...)
            except ImportError as exc:
                _log.error("isaac_sim_driver_app_launcher_failed", error=str(exc))
                msg = (
                    f"isaac extras not installed: {exc}. "
                    'Install with `pip install -e ".[isaac]" '
                    "--extra-index-url https://pypi.nvidia.com`."
                )
                raise ArmDriverError(msg) from exc

            # Kit boot MUST run on the main thread of the main interpreter.
            # ``isaacsim.simulation_app.SimulationApp.__init__`` calls
            # ``signal.signal(SIGINT, ...)`` which Python only allows from
            # the main thread; ``asyncio.to_thread`` schedules work on the
            # default ThreadPoolExecutor and trips
            # ``ValueError: signal only works in main thread of the main
            # interpreter``. The bootstrap is a one-shot per process
            # (Kit's AppLauncher singleton, see _app_state) so the
            # ~10-20 s blocking call against the event loop is acceptable
            # — runtime motion still uses ``asyncio.to_thread`` for
            # ``set_joint_position_target`` / ``write_data_to_sim`` /
            # the ``_step_sim`` loop, where the main-thread requirement
            # does not apply.
            self._launch_kit_and_articulation()
            _app_state.mark_launched()
            self._connected = True
            _log.info(
                "isaac_sim_driver_connected",
                n_dofs=self._dof,
                n_articulations=1,
            )

    def _launch_kit_and_articulation(self) -> None:
        """Synchronous Kit boot + SimulationContext + runtime Articulation construction.

        Called directly from ``connect()`` (NOT via ``asyncio.to_thread``)
        because Isaac Sim's ``SimulationApp.__init__`` registers a
        ``SIGINT`` handler via ``signal.signal``, which Python's signal
        module rejects when called off the main thread of the main
        interpreter. ``connect()`` awaits this method synchronously; the
        event loop pause for the ~10-20 s Kit bootstrap is acceptable
        because Kit is a process-singleton (see ``_app_state``) and the
        boot fires at most once per process lifetime. Runtime motion
        APIs (``set_joint_position_target`` / ``write_data_to_sim`` /
        ``_step_sim``) still use ``asyncio.to_thread`` because that
        constraint applies only to ``signal.signal`` registration.

        Sequence (Isaac Lab 2.3):
          0. (Caller) ``connect()`` invokes this on the main thread —
             see the docstring above for the SIGINT-on-main-thread
             reasoning.
          1. ``AppLauncher`` boots Kit + Omniverse extensions.
          2. ``SimulationContext`` creates the physics scene (must
             happen AFTER Kit; ``isaaclab.sim`` cannot be imported
             until Kit is live).
          3. Minimal stage scaffolding (ground plane + dome light) is
             spawned — required for the physics step to converge.
          4. The articulation prim is spawned at ``/World/SO_ARM100``.
          5. ``Articulation(cfg)`` wraps the spawned prim with the
             runtime API (``set_joint_position_target``,
             ``write_data_to_sim``, ``.data.joint_pos``).
          6. ``sim.reset()`` commits the spawn so the runtime asset is
             queryable.

        PR-11 review fix C1 (gemini #1, #2, copilot #11): previously
        this method stored the ``ArticulationCfg`` in
        ``self._articulation`` instead of instantiating the runtime
        ``Articulation`` — every subsequent call to
        ``set_joint_position_target`` / ``write_data_to_sim`` /
        ``.data.*`` would have raised ``AttributeError``.
        """
        from isaaclab.app import AppLauncher

        self._app_launcher = AppLauncher(
            launcher_args={
                "headless": self._sim_cfg.headless,
                "enable_cameras": self._sim_cfg.enable_cameras,
            }
        )

        # Kit-dependent imports MUST happen after AppLauncher boots —
        # isaaclab.sim wires into omni.kit on import.
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab.sim import SimulationCfg, SimulationContext

        from armdroid.hardware.isaac_sim.articulation import (
            build_so_arm100_articulation_cfg,
        )

        sim_cfg = SimulationCfg(
            dt=self._sim_cfg.physics_dt_s,
            render_interval=self._sim_cfg.decimation,
        )
        self._sim_context = SimulationContext(sim_cfg)

        # Minimal stage scaffolding — physics step needs a ground plane;
        # the dome light only matters when not headless but is cheap.
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/light", light_cfg)

        # Build the cfg, attach the prim path, then instantiate the
        # runtime Articulation. The runtime object exposes the joint
        # API the rest of this driver depends on.
        articulation_cfg = build_so_arm100_articulation_cfg(self._sim_cfg, self._cfg)
        articulation_cfg = articulation_cfg.replace(prim_path="/World/SO_ARM100")
        self._articulation = Articulation(cfg=articulation_cfg)

        # Commit the spawn so the runtime asset is queryable on first
        # send_joint_positions / read_state call.
        self._sim_context.reset()

    def _step_sim(self, n_steps: int) -> None:
        """Advance physics ``n_steps`` times, refreshing articulation cache.

        Synchronous helper — callers wrap in ``asyncio.to_thread`` so the
        Kit step loop does not block the event loop.
        """
        for _ in range(n_steps):
            self._sim_context.step(render=not self._sim_cfg.headless)
            self._articulation.update(self._sim_cfg.physics_dt_s)

    async def disconnect(self) -> None:
        """Close Kit + release the articulation. Idempotent."""
        if not self._connected:
            return
        with get_telemetry().start_span(SPAN_DRIVER_DISCONNECT):
            _log.info("isaac_sim_driver_disconnect")
            if self._app_launcher is not None and hasattr(self._app_launcher, "app"):
                await asyncio.to_thread(self._app_launcher.app.close)
            self._connected = False
            _log.info("isaac_sim_driver_disconnected")

    # ------------------------------------------------------------------ #
    # Modern motion
    # ------------------------------------------------------------------ #

    async def send_joint_positions(
        self,
        positions: tuple[float, ...],
        duration_s: float,
    ) -> None:
        """Command an interpolated move to ``positions`` over ``duration_s``."""
        if self._estop_latched:
            msg = "emergency stop latched; clear via clear_emergency_stop()"
            raise ArmCommandRejected(msg)
        if not self._connected:
            msg = "driver not connected; call connect() first"
            raise ArmDriverError(msg)
        if len(positions) != self._dof:
            msg = f"positions length {len(positions)} != dof {self._dof}"
            raise ArmCommandRejected(msg)
        if duration_s <= 0.0:
            msg = f"duration_s must be > 0; got {duration_s}"
            raise ArmCommandRejected(msg)
        gripper_idx = self._sim_cfg.gripper_joint_index
        if not (0 <= gripper_idx < self._dof):
            # Pydantic schema clamps gripper_joint_index to [0, 11], but
            # nothing reconciles it against cfg.dof. A misconfig (dof=6,
            # gripper_index=8) would otherwise raise IndexError deep in
            # normalised_vector_to_radians; reject up front instead.
            msg = (
                f"sim_cfg.gripper_joint_index={gripper_idx} out of range "
                f"for dof={self._dof}; must satisfy 0 <= idx < dof."
            )
            raise ArmCommandRejected(msg)
        for i, p in enumerate(positions):
            if not np.isfinite(p):
                msg = f"position[{i}]={p} is not finite"
                raise ArmCommandRejected(msg)
            if i == gripper_idx:
                # Gripper inputs are armdroid-normalised (0=open, 1=closed
                # by default). Validate against the configured normalised
                # window so a stray scale (e.g. callers passing radians)
                # is caught here rather than producing unphysical commands
                # after normalised_vector_to_radians.
                lo = min(
                    self._sim_cfg.gripper_normalised_open,
                    self._sim_cfg.gripper_normalised_closed,
                )
                hi = max(
                    self._sim_cfg.gripper_normalised_open,
                    self._sim_cfg.gripper_normalised_closed,
                )
                if not (lo <= p <= hi):
                    msg = f"gripper position[{i}]={p} outside normalised window [{lo}, {hi}]."
                    raise ArmCommandRejected(msg)
            else:
                limits = self._cfg.joint_limits[i]
                if not (limits.min_rad <= p <= limits.max_rad):
                    msg = (
                        f"position[{i}]={p} rad outside joint limits "
                        f"[{limits.min_rad}, {limits.max_rad}]."
                    )
                    raise ArmCommandRejected(msg)

        with get_telemetry().start_span(
            SPAN_DRIVER_SEND,
            dof=self._dof,
            duration_s=duration_s,
        ):
            _log.debug(
                "isaac_sim_driver_send",
                dof=self._dof,
                duration_s=duration_s,
            )
            try:
                # Convert gripper from armdroid normalised → URDF radians.
                positions_rad = normalised_vector_to_radians(
                    positions,
                    self._sim_cfg.gripper_joint_index,
                    self._sim_cfg,
                )
                # Use Isaac Lab's Articulation singular API (verified Isaac Lab 2.3).
                import torch

                target = torch.tensor(positions_rad, dtype=torch.float32).reshape(
                    self._sim_cfg.num_envs, -1
                )
                await asyncio.to_thread(
                    self._articulation.set_joint_position_target,
                    target,
                )
                await asyncio.to_thread(self._articulation.write_data_to_sim)
                # Advance the sim so the PD controller actually executes
                # the move; without stepping, set_joint_position_target +
                # write_data_to_sim leave the articulation state unchanged.
                n_steps = max(1, round(duration_s / self._sim_cfg.physics_dt_s))
                await asyncio.to_thread(self._step_sim, n_steps)
            except Exception as exc:
                _log.error(
                    "isaac_sim_driver_send_failed",
                    error=str(exc),
                    exc_info=True,
                )
                raise

    async def read_state(self) -> ArmState:
        """Return the latest cached arm telemetry from Isaac Lab's Articulation.data."""
        if not self._connected:
            msg = "driver not connected; call connect() first"
            raise ArmDriverError(msg)
        _log.debug("isaac_sim_driver_read_state", n_joints=self._dof)
        # Pull joint state from articulation.data; convert gripper back to normalised.
        joint_pos_rad = tuple(float(v) for v in self._articulation.data.joint_pos[0].cpu().numpy())
        joint_vel = tuple(float(v) for v in self._articulation.data.joint_vel[0].cpu().numpy())
        import time

        joint_pos_norm = radians_vector_to_normalised(
            joint_pos_rad,
            self._sim_cfg.gripper_joint_index,
            self._sim_cfg,
        )
        return ArmState(
            joint_positions=joint_pos_norm,
            joint_velocities=joint_vel,
            is_moving=any(abs(v) > 1e-6 for v in joint_vel),
            estop_active=self._estop_latched,
            timestamp_s=time.monotonic(),
        )

    # ------------------------------------------------------------------ #
    # Modern safety
    # ------------------------------------------------------------------ #

    async def emergency_stop(self) -> None:
        """Latch firmware (or sim) into emergency stop."""
        self._estop_latched = True
        _log.warning("isaac_sim_driver_estop_latched", connected=self._connected)
        if self._connected and self._articulation is not None:
            # Apply zero target to freeze the articulation.
            import torch

            zero = torch.zeros((self._sim_cfg.num_envs, self._dof), dtype=torch.float32)
            await asyncio.to_thread(
                self._articulation.set_joint_position_target,
                zero,
            )
            # Push the zero target to sim and step once so the e-stop
            # actually takes effect rather than sitting in the staged
            # buffer until the next send_joint_positions().
            await asyncio.to_thread(self._articulation.write_data_to_sim)
            await asyncio.to_thread(self._step_sim, 1)

    async def clear_emergency_stop(self) -> None:
        """Release the e-stop latch."""
        self._estop_latched = False
        _log.info("isaac_sim_driver_estop_cleared")

    # ------------------------------------------------------------------ #
    # Legacy adapters — preserved for backwards compatibility
    # ------------------------------------------------------------------ #

    async def get_joint_states(self) -> NDArray[np.float64]:
        """Read current joint angles as a numpy array (legacy)."""
        state = await self.read_state()
        return np.asarray(state.joint_positions, dtype=np.float64)

    async def send_joint_command(self, target_angles: NDArray[np.float64]) -> None:
        """Command target joint angles (legacy; no duration control)."""
        await self.send_joint_positions(tuple(float(v) for v in target_angles), duration_s=1.0)

    async def close_gripper(self) -> float:
        """Close the gripper (legacy). Returns the grip force in Newtons."""
        state = await self.read_state()
        positions = list(state.joint_positions)
        positions[self._sim_cfg.gripper_joint_index] = self._sim_cfg.gripper_normalised_closed
        await self.send_joint_positions(tuple(positions), duration_s=1.0)
        return self._sim_cfg.gripper_effort_limit_sim

    async def open_gripper(self) -> None:
        """Open the gripper fully (legacy)."""
        state = await self.read_state()
        positions = list(state.joint_positions)
        positions[self._sim_cfg.gripper_joint_index] = self._sim_cfg.gripper_normalised_open
        await self.send_joint_positions(tuple(positions), duration_s=1.0)

    async def home(self) -> None:
        """Move arm to home position (legacy)."""
        home = tuple(float(v) for v in self._cfg.home_position)
        await self.send_joint_positions(home, duration_s=self._cfg.home_duration_s)

    async def start(self) -> None:
        """Open the transport (legacy alias for :meth:`connect`)."""
        await self.connect()

    async def stop(self) -> None:
        """Close the transport (legacy alias for :meth:`disconnect`)."""
        await self.disconnect()


__all__ = ["IsaacSimDriver"]
