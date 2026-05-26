"""Runtime-checkable Protocol contracts spoken by armdroid layers.

All arm component interfaces use ``@runtime_checkable`` structural typing.
Concrete implementations live under :mod:`armdroid.hardware`,
:mod:`armdroid.perception`, :mod:`armdroid.planning`, :mod:`armdroid.control`,
and :mod:`armdroid.environments`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from armdroid.domain.state import (
    ArmAction,
    ArmState,
    DetectedObject,
    InteractionEvent,
    PlanStep,
    SceneInsight,
    SymbolicState,
    Verdict,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

    import numpy as np
    import torch
    from numpy.typing import NDArray


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

    def build_for_env(
        self,
        env: ArmEnvironmentProtocol | VecArmEnvironmentProtocol,
    ) -> None:
        """Bind the internal RL model to an environment (lazy wiring).

        Safe to call multiple times - only builds if not already built.
        Accepts either a single-env or vec env; concrete controllers
        dispatch internally based on the runtime protocol type.

        Args:
            env: Gymnasium-compatible environment for the configured
                training algorithm.
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


@runtime_checkable
class ArmRLAgentProtocol(Protocol):
    """Common contract for RL agents (SAC, RSL-RL PPO, ...).

    Used by the ``armdroid.rl_agents`` registry so the orchestrator can
    dispatch to any registered agent uniformly. Concrete implementations
    live under :mod:`armdroid.control` (``SACAgent`` today; PR-B adds
    ``RslRlPpoAgent``).
    """

    def build(self, env: ArmEnvironmentProtocol) -> None:
        """Bind the underlying model to ``env``. Idempotent.

        Args:
            env: Gymnasium-compatible environment.
        """
        ...

    def train(self, total_timesteps: int | None = None) -> None:
        """Train for ``total_timesteps`` (or use the config default).

        Args:
            total_timesteps: Optional override; ``None`` uses the agent's
                configuration's default.
        """
        ...

    def predict(
        self,
        observation: dict[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Return greedy action for ``observation``.

        Args:
            observation: Goal-conditioned observation dict.

        Returns:
            Action vector.
        """
        ...

    def save(self, path: str | None = None) -> Path:
        """Save model to ``path`` (or the config default).

        Args:
            path: Optional override; ``None`` uses the agent's
                configured ``weights_dir``.

        Returns:
            Filesystem path of the saved checkpoint.
        """
        ...

    def load(self, path: str) -> None:
        """Load model from ``path``.

        Args:
            path: Path to a saved checkpoint.
        """
        ...

    @property
    def is_trained(self) -> bool:
        """Whether ``train()`` has completed at least once."""
        ...

    @property
    def is_built(self) -> bool:
        """Whether ``build()`` has bound an environment."""
        ...


@runtime_checkable
class VecArmEnvironmentProtocol(Protocol):
    """Vectorised environment interface for parallel training (``num_envs > 1``).

    Sibling to :class:`ArmEnvironmentProtocol`. Returns batched torch
    tensors rather than the per-step numpy scalars of the single-env
    protocol. The companion :meth:`as_runner_env` accessor exposes the
    underlying RL-runner-compatible env (typically Isaac Lab's
    ``ManagerBasedRLEnv``) without forcing consumers to reach through
    private attributes on the env wrapper.

    Permits ``num_envs >= 1`` so callers may adopt the vec protocol
    unconditionally; the orchestration factory still routes
    ``num_envs == 1`` to the single-env path by default.
    """

    @property
    def num_envs(self) -> int:
        """Number of parallel environments. Must be ``>= 1``."""
        ...

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Reset all parallel envs.

        Returns:
            Tuple ``(obs_dict, info)``. Every value in ``obs_dict`` has
            leading dim equal to :attr:`num_envs`.
        """
        ...

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Step all parallel envs.

        Args:
            action: Tensor of shape ``(num_envs, action_dim)``.

        Returns:
            Tuple ``(obs, reward, terminated, truncated, info)`` with
            ``reward`` / ``terminated`` / ``truncated`` each shape
            ``(num_envs,)``.
        """
        ...

    def close(self) -> None:
        """Release all parallel envs."""
        ...

    def as_runner_env(self) -> Any:
        """Return the underlying RL-runner-compatible env.

        Replacement for the legacy ``env._isaac_env`` reach-through.
        Returns the raw ``ManagerBasedRLEnv`` (or analogue) so RL-runner
        backends (RSL-RL's ``OnPolicyRunner``) can consume it directly.
        """
        ...


@runtime_checkable
class VecArmRLAgentProtocol(Protocol):
    """RL agent contract for vectorised training.

    Sibling to :class:`ArmRLAgentProtocol`. Agents may implement both
    surfaces (single-env via ``build`` / ``train`` and vec via
    ``build_vec`` / ``train_vec``); the orchestration layer picks the
    right call site based on the env type and
    ``cfg.arm_sim_isaac.num_envs``.
    """

    def build_vec(self, env: VecArmEnvironmentProtocol) -> None:
        """Build the underlying runner around a vec env. Idempotent."""
        ...

    def train_vec(self, total_timesteps: int | None = None) -> None:
        """Run vectorised training to completion."""
        ...

    def predict(
        self,
        observation: dict[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Single-step inference. Inputs / outputs stay numpy at the boundary."""
        ...

    def save(self, path: str | None = None) -> Path:
        """Persist trained weights."""
        ...

    def load(self, path: str) -> None:
        """Load trained weights."""
        ...

    @property
    def is_trained(self) -> bool:
        """Whether ``train_vec`` has completed at least once."""
        ...

    @property
    def is_built(self) -> bool:
        """Whether ``build_vec`` has bound a runner."""
        ...


@runtime_checkable
class VisionLanguageAgentProtocol(Protocol):
    """Vision-Language-Action agent contract (Gemini ER 1.6 and successors).

    Sibling to :class:`ArmRLAgentProtocol` per ADR-0007. Methods are
    intentionally disjoint from the RL surface (``build``, ``train``,
    ``predict``, ``save``, ``load``, ``is_built``, ``is_trained``) so
    runtime ``isinstance`` dispatch is unambiguous. See ADR-0007 for the
    rationale.

    A VLA agent consumes raw observations (RGB + depth + natural-language
    instruction + arm state) and emits :class:`ArmAction` directly; the
    pixels never pass through a symbolic state extractor.
    """

    def build_vla(self, env: ArmEnvironmentProtocol) -> None:
        """Bind the VLA backend to ``env``. Idempotent.

        Args:
            env: Gymnasium-compatible environment whose action space the
                VLA's output must satisfy. Used to validate joint count
                and action ranges at build time.
        """
        ...

    def act(
        self,
        rgb: NDArray[np.uint8],
        depth: NDArray[np.float32],
        instruction: str,
        state: ArmState,
    ) -> ArmAction:
        """Produce one action for the given multi-modal observation.

        Args:
            rgb: Current RGB frame, shape ``(H, W, 3)``.
            depth: Current depth frame, shape ``(H, W)`` (metres).
            instruction: Natural-language task instruction.
            state: Latest :class:`ArmState` snapshot from the driver.

        Returns:
            :class:`ArmAction` whose ``joint_targets`` length matches the
            arm's degrees of freedom.
        """
        ...

    @property
    def is_vla_built(self) -> bool:
        """Whether :meth:`build_vla` has bound an environment."""
        ...

    @property
    def is_vla_loaded(self) -> bool:
        """Whether the underlying VLA weights / SDK handle are ready."""
        ...


@runtime_checkable
class HighLevelPlannerProtocol(Protocol):
    """Foundation-model high-level planner contract (e.g. Gemini ER 1.6).

    Takes a natural-language goal plus optional scene context and emits
    a sequence of :class:`PlanStep` items that the standard PDDL
    executor can run. Disjoint from :class:`ArmPlannerProtocol` because
    inputs and outputs are different: a high-level planner accepts free
    text, not a symbolic state.
    """

    @property
    def name(self) -> str:
        """Backend name (``"gemini_er"`` etc.) for logging and registry lookup."""
        ...

    def plan_high_level(
        self,
        goal_text: str,
        scene: SceneInsight | None = None,
    ) -> list[PlanStep]:
        """Translate a natural-language goal into ordered :class:`PlanStep`s.

        Args:
            goal_text: Free-form task description (e.g. "put the blue
                block in the orange bowl").
            scene: Optional :class:`SceneInsight` providing grounded
                visual context (crops, detections, notes). When ``None``
                the planner relies on the goal text alone.

        Returns:
            Ordered list of :class:`PlanStep` items. Empty list is the
            documented "give up" signal so callers can fall back to a
            symbolic planner without raising.
        """
        ...


@runtime_checkable
class SafetyGuardProtocol(Protocol):
    """Single layer in the Swiss-cheese safety chain (ADR-0009).

    A guard either approves or rejects a candidate plan or action. The
    chain composer enforces deny-overrides semantics so any single
    ``allowed=False`` verdict short-circuits the chain.
    """

    @property
    def name(self) -> str:
        """Guard name (``"iso_kinematic"``, ``"asimov"`` etc.) for logs."""
        ...

    def check_plan(self, steps: list[PlanStep]) -> Verdict:
        """Evaluate a candidate plan against the guard's rules.

        Args:
            steps: Ordered plan steps from a planner or replanner.

        Returns:
            :class:`Verdict` with ``allowed=True`` to approve, or
            ``allowed=False`` plus a populated ``reason``.
        """
        ...

    def check_action(self, action: ArmAction, state: ArmState) -> Verdict:
        """Evaluate a single action against the guard's rules.

        Args:
            action: :class:`ArmAction` produced by the controller.
            state: Current :class:`ArmState` to inform the check
                (velocity, position, e-stop state).

        Returns:
            :class:`Verdict` matching :meth:`check_plan` semantics.
        """
        ...


@runtime_checkable
class InteractionSessionProtocol(Protocol):
    """Bidirectional audio/video interaction session (Gemini Live API).

    Implementations act as async context managers so the session
    lifecycle is bounded by ``async with``. The orchestrator awaits
    :meth:`receive_events` on a separate task and forwards
    ``InteractionEvent`` items to the replanner.
    """

    async def __aenter__(self) -> InteractionSessionProtocol:
        """Open the underlying transport (e.g. Gemini Live websocket)."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the underlying transport. Idempotent."""
        ...

    async def send_audio(self, chunk: bytes) -> None:
        """Forward a PCM audio chunk upstream."""
        ...

    async def send_frame(self, rgb: NDArray[np.uint8]) -> None:
        """Forward an RGB frame upstream."""
        ...

    def receive_events(self) -> AsyncIterator[InteractionEvent]:
        """Async iterator of events emitted by the remote service."""
        ...


__all__ = [
    "ArmControllerProtocol",
    "ArmDriverProtocol",
    "ArmEnvironmentProtocol",
    "ArmPerceptionProtocol",
    "ArmPlannerProtocol",
    "ArmRLAgentProtocol",
    "HighLevelPlannerProtocol",
    "InteractionSessionProtocol",
    "SafetyGuardProtocol",
    "VecArmEnvironmentProtocol",
    "VecArmRLAgentProtocol",
    "VisionLanguageAgentProtocol",
]
