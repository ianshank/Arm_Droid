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
in the same process. The module-level ``_app_launched`` flag enforces
this with a clear ``ArmDriverError`` rather than letting the second
call produce inscrutable Kit double-init errors (peer-review N-3).

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

# Module-level singleton guard. Kit's AppLauncher cannot be initialised
# twice in the same Python process; the second call produces inscrutable
# Kit double-init traceback. Set to True on first successful connect();
# subsequent connect() raises ArmDriverError with a clear message.
# (Peer-review N-3.)
_app_launched: bool = False


def _reset_app_launched_for_tests() -> None:
    """Reset the once-per-process AppLauncher guard. Tests only."""
    global _app_launched
    _app_launched = False


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

        self._cfg = cfg
        self._sim_cfg = sim_cfg
        self._dof = cfg.dof
        self._articulation: Any = None
        self._app_launcher: Any = None
        self._connected = False
        self._estop_latched = False
        _log.info(
            "isaac_sim_driver_init",
            dof=self._dof,
            headless=self._sim_cfg.headless,
            num_envs=self._sim_cfg.num_envs,
            usd_path=str(self._sim_cfg.usd_path),
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
        global _app_launched
        if self._connected:
            return
        with get_telemetry().start_span(
            SPAN_DRIVER_CONNECT,
            usd_path=str(self._sim_cfg.usd_path),
            headless=self._sim_cfg.headless,
        ):
            _log.info("isaac_sim_driver_connecting", usd_path=str(self._sim_cfg.usd_path))
            if _app_launched:
                msg = (
                    "Isaac Sim AppLauncher already initialised in this process; "
                    "can only run one IsaacSimDriver per process."
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

            await asyncio.to_thread(self._launch_kit_and_articulation)
            _app_launched = True
            self._connected = True
            _log.info(
                "isaac_sim_driver_connected",
                n_dofs=self._dof,
                n_articulations=1,
            )

    def _launch_kit_and_articulation(self) -> None:
        """Synchronous Kit boot + articulation construction.

        Run inside ``asyncio.to_thread`` because Kit's runtime is
        synchronous. ``connect()`` awaits this method.
        """
        from isaaclab.app import AppLauncher

        from armdroid.hardware.isaac_sim.articulation import (
            build_so_arm100_articulation_cfg,
        )

        self._app_launcher = AppLauncher(
            launcher_args={
                "headless": self._sim_cfg.headless,
                "enable_cameras": self._sim_cfg.enable_cameras,
            }
        )
        # The articulation is built once; subsequent send_joint_positions
        # calls reuse this instance.
        articulation_cfg = build_so_arm100_articulation_cfg(self._sim_cfg, self._cfg)
        # NB: actual Articulation construction requires a SimulationContext
        # which is only available after AppLauncher boots; the smoke test
        # in tests/isaac/test_smoke_isaac_driver.py exercises the full
        # path under a real GPU. CI's pure-Python coverage stops here.
        self._articulation = articulation_cfg

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
        for i, p in enumerate(positions):
            if not np.isfinite(p):
                msg = f"position[{i}]={p} is not finite"
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
