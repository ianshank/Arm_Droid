"""Runtime-checkable Protocol contracts spoken by armdroid layers.

All arm component interfaces use ``@runtime_checkable`` structural typing.
Concrete implementations live under :mod:`armdroid.hardware`,
:mod:`armdroid.perception`, :mod:`armdroid.planning`, :mod:`armdroid.control`,
and :mod:`armdroid.environments`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from armdroid.domain.state import ArmState, DetectedObject, PlanStep, SymbolicState


@runtime_checkable
class ArmDriverProtocol(Protocol):
    """Interface for robot arm hardware drivers (mock, ESP32-JSON).

    Two surfaces are exposed:

    * **Modern** lifecycle (``connect`` / ``disconnect`` / ``is_connected``),
      latched-e-stop (``emergency_stop`` / ``clear_emergency_stop``), and
      interpolated motion (``send_joint_positions``, ``read_state``). New
      code should target this surface.
    * **Legacy** adapters (``start`` / ``stop`` / ``send_joint_command`` /
      ``get_joint_states`` / ``open_gripper`` / ``close_gripper`` /
      ``home``) preserved so existing controllers and tests run unchanged.
    """

    # ---- Modern lifecycle ------------------------------------------------

    async def connect(self) -> None:
        """Open the transport. Idempotent.

        Raises:
            ArmDriverError: when the transport cannot be opened.
        """
        ...

    async def disconnect(self) -> None:
        """Close the transport and release resources. Idempotent."""
        ...

    @property
    def is_connected(self) -> bool:
        """Whether the transport is currently open."""
        ...

    # ---- Modern motion ---------------------------------------------------

    async def send_joint_positions(
        self,
        positions: tuple[float, ...],
        duration_s: float,
    ) -> None:
        """Command an interpolated move to ``positions`` over ``duration_s``.

        Args:
            positions: Target joint values. Length must equal :attr:`dof`.
                Units: radians for rotational joints, normalised ``[0, 1]``
                for the gripper joint (when present).
            duration_s: Time over which the firmware (or mock) interpolates
                from the current pose to ``positions``. Must be positive.

        Raises:
            ArmCommandRejected: bad shape, NaN/inf, joint-limit violation,
                velocity-limit violation, or e-stop latched.
            ArmDriverError: transport-layer failure.
        """
        ...

    async def read_state(self) -> ArmState:
        """Return the latest cached arm telemetry.

        Returns:
            :class:`ArmState` snapshot. Implementations should cache the
            most recent firmware heartbeat and return it without blocking.
        """
        ...

    # ---- Modern safety ---------------------------------------------------

    async def emergency_stop(self) -> None:
        """Latch firmware (or mock) into emergency stop.

        All subsequent motion commands are rejected with
        :class:`ArmCommandRejected` until :meth:`clear_emergency_stop` is
        called. Implementations make a best-effort attempt to deliver the
        e-stop command, but may raise :class:`ArmDriverError` if the driver
        is not connected or if the underlying write fails.
        """
        ...

    async def clear_emergency_stop(self) -> None:
        """Release the e-stop latch. The arm remains stationary."""
        ...

    # ---- Legacy adapters (preserved for backwards compatibility) ---------

    async def get_joint_states(self) -> NDArray[np.float64]:
        """Read current joint angles as a numpy array (legacy)."""
        ...

    async def send_joint_command(self, target_angles: NDArray[np.float64]) -> None:
        """Command target joint angles (legacy; no duration control)."""
        ...

    async def close_gripper(self) -> float:
        """Close the gripper (legacy). Returns the grip force in Newtons."""
        ...

    async def open_gripper(self) -> None:
        """Open the gripper fully (legacy)."""
        ...

    async def home(self) -> None:
        """Move arm to home position (legacy)."""
        ...

    async def start(self) -> None:
        """Open the transport (legacy alias for :meth:`connect`)."""
        ...

    async def stop(self) -> None:
        """Close the transport (legacy alias for :meth:`disconnect`)."""
        ...

    @property
    def dof(self) -> int:
        """Degrees of freedom (joint vector length)."""
        ...


@runtime_checkable
class ArmPerceptionProtocol(Protocol):
    """Interface for robot arm perception stack (Layer 0)."""

    async def detect_objects(self) -> list[DetectedObject]:
        """Detect objects in current camera frame.

        Returns:
            List of detected objects with poses.
        """
        ...

    async def get_symbolic_state(self) -> SymbolicState:
        """Convert current detections to symbolic PDDL state.

        Returns:
            Symbolic state with active predicates.
        """
        ...

    async def start(self) -> None:
        """Start perception pipeline."""
        ...

    async def stop(self) -> None:
        """Stop perception pipeline."""
        ...


@runtime_checkable
class ArmPlannerProtocol(Protocol):
    """Interface for symbolic planning (Layer 1)."""

    def plan(self, initial_state: SymbolicState, goal_state: SymbolicState) -> list[PlanStep]:
        """Generate optimal plan from initial to goal state.

        Args:
            initial_state: Current symbolic state.
            goal_state: Target symbolic state.

        Returns:
            Ordered list of plan steps.

        Raises:
            PlanningError: If no valid plan exists.
        """
        ...

    def replan(
        self, current_state: SymbolicState, goal_state: SymbolicState, error: str
    ) -> list[PlanStep]:
        """Generate recovery plan after execution failure.

        Args:
            current_state: Current (possibly unexpected) symbolic state.
            goal_state: Original target state.
            error: Description of the failure.

        Returns:
            Recovery plan steps.
        """
        ...


@runtime_checkable
class ArmControllerProtocol(Protocol):
    """Interface for low-level motor control (Layer 3)."""

    async def execute_action(self, action: NDArray[np.float64]) -> dict[str, Any]:
        """Execute continuous action (joint deltas or end-effector target).

        Args:
            action: Action vector from RL policy.

        Returns:
            Execution result with keys: success, achieved_state, info.
        """
        ...

    async def execute_primitive(self, primitive_name: str, target: NDArray[np.float64]) -> bool:
        """Execute pre-trained action primitive.

        Args:
            primitive_name: Primitive name ("grasp", "place", "transit").
            target: Target pose for the primitive.

        Returns:
            True if primitive executed successfully.
        """
        ...

    def build_for_env(self, env: ArmEnvironmentProtocol) -> None:
        """Bind the internal RL model to an environment (lazy wiring).

        Safe to call multiple times — only builds if not already built.

        Args:
            env: Gymnasium-compatible environment for SAC+HER.
        """
        ...

    def train_policy(self, total_timesteps: int | None = None) -> Path:
        """Train the internal RL policy and save a checkpoint.

        Args:
            total_timesteps: Override config total_timesteps (None = use config).

        Returns:
            Filesystem path of the saved policy checkpoint.
        """
        ...


@runtime_checkable
class ArmEnvironmentProtocol(Protocol):
    """Interface for Gymnasium-compatible arm training environments."""

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Optional random seed.

        Returns:
            Tuple of (observation, info).
        """
        ...

    def step(
        self, action: NDArray[np.float64]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Action to execute.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        ...

    def render(self) -> NDArray[np.uint8] | None:
        """Render current state.

        Returns:
            RGB image array or None if rendering disabled.
        """
        ...

    def close(self) -> None:
        """Clean up environment resources."""
        ...


__all__ = [
    "ArmControllerProtocol",
    "ArmDriverProtocol",
    "ArmEnvironmentProtocol",
    "ArmPerceptionProtocol",
    "ArmPlannerProtocol",
]
