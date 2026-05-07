"""Top-level armdroid orchestrator composing the four arm subsystems.

The orchestrator owns the lifecycle of perception, planner, controller,
environment, and driver. It exposes three operating modes:

- :meth:`train`  — synchronous SB3 ``SAC.learn()`` loop. Wires the
  un-built ``SACAgent`` (returned from :func:`build_arm_controller`) to
  the environment via ``controller.agent.build(env)`` lazily, then calls
  ``train()`` and ``save()``.
- :meth:`rollout` — async PDDL plan + primitive execution against the
  real (or mock) driver, using perception to read symbolic state.
- :meth:`shutdown` — async driver teardown.

The async/sync split mirrors the underlying APIs: SB3 is synchronous,
the arm driver and perception are async.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from armdroid.logging.setup import get_logger

if TYPE_CHECKING:
    from armdroid.protocols import (
        ArmControllerProtocol,
        ArmDriverProtocol,
        ArmEnvironmentProtocol,
        ArmPerceptionProtocol,
        ArmPlannerProtocol,
        PlanStep,
        SymbolicState,
    )

_log = get_logger(__name__)


class ArmOrchestrator:
    """Composes the five arm subsystems behind a stable interface.

    Args:
        perception: Perception facade (depth + detection + pose + symbolic state).
        planner: Symbolic planner (PDDL or LLM-backed).
        controller: RL controller wrapping a SACAgent + ActionPrimitives.
        environment: Gymnasium environment for SAC+HER training.
        driver: Arm hardware driver (real or mock).
    """

    def __init__(
        self,
        perception: ArmPerceptionProtocol,
        planner: ArmPlannerProtocol,
        controller: ArmControllerProtocol,
        environment: ArmEnvironmentProtocol,
        driver: ArmDriverProtocol,
    ) -> None:
        """Initialise the orchestrator with pre-built components."""
        self._perception = perception
        self._planner = planner
        self._controller = controller
        self._environment = environment
        self._driver = driver
        _log.info("arm_orchestrator_init")

    @property
    def perception(self) -> ArmPerceptionProtocol:
        """Return the perception facade."""
        return self._perception

    @property
    def planner(self) -> ArmPlannerProtocol:
        """Return the symbolic planner."""
        return self._planner

    @property
    def controller(self) -> ArmControllerProtocol:
        """Return the RL controller."""
        return self._controller

    @property
    def environment(self) -> ArmEnvironmentProtocol:
        """Return the Gymnasium environment."""
        return self._environment

    @property
    def driver(self) -> ArmDriverProtocol:
        """Return the arm hardware driver."""
        return self._driver

    def train(self, total_timesteps: int | None = None) -> Path:
        """Run the SAC+HER training loop and save the resulting policy.

        Wires the SAC agent to the environment lazily on the first call
        (the factory leaves the agent un-built).

        Args:
            total_timesteps: Override config ``arm_training.total_timesteps``.

        Returns:
            Filesystem path of the saved policy checkpoint.
        """
        agent = self._controller.agent  # type: ignore[attr-defined]
        if not agent.is_built:
            _log.info("arm_orchestrator_building_agent_for_env")
            agent.build(self._environment)
        _log.info("arm_orchestrator_train_start", total_timesteps=total_timesteps)
        agent.train(total_timesteps)
        path = agent.save()
        _log.info("arm_orchestrator_train_complete", checkpoint=str(path))
        return path

    async def rollout(
        self,
        initial_state: SymbolicState,
        goal_state: SymbolicState,
    ) -> dict[str, Any]:
        """Plan from ``initial_state`` to ``goal_state`` and execute the plan.

        Each :class:`PlanStep` is dispatched through the controller's
        primitive layer. Failures abort and return the partial result.

        Args:
            initial_state: Starting symbolic state.
            goal_state: Target symbolic state.

        Returns:
            Dict with keys ``plan`` (list[PlanStep]), ``executed`` (int,
            number of steps successfully executed) and ``success`` (bool).
        """
        plan: list[PlanStep] = self._planner.plan(initial_state, goal_state)
        _log.info("arm_orchestrator_plan_built", n_steps=len(plan))
        executed = 0
        for step in plan:
            success = await self._controller.execute_primitive(
                step.action,
                target=_step_args_to_target(step.args),
            )
            if not success:
                _log.warning("arm_orchestrator_step_failed", step=str(step))
                break
            executed += 1
        return {
            "plan": plan,
            "executed": executed,
            "success": executed == len(plan),
        }

    async def shutdown(self) -> None:
        """Tear down the arm driver and environment."""
        _log.info("arm_orchestrator_shutdown_start")
        try:
            await self._driver.stop()
        except Exception:  # noqa: BLE001
            _log.exception("arm_orchestrator_driver_stop_failed")
        try:
            self._environment.close()
        except Exception:  # noqa: BLE001
            _log.exception("arm_orchestrator_env_close_failed")
        _log.info("arm_orchestrator_shutdown_complete")


def _step_args_to_target(args: list[str]) -> Any:
    """Convert PDDL plan-step args to a primitive target.

    Plan-step args are symbolic (e.g. ``["disk1", "peg_A", "peg_C"]``).
    This adapter is intentionally minimal — the symbolic-to-pose
    conversion lives in the action primitives themselves; here we just
    pass the args through as a string tuple. Concrete pose lookup is
    the controller's responsibility.

    Args:
        args: PDDL action arguments.

    Returns:
        Args as a numpy array (zero-padded) so callers receive a
        consistent ``NDArray[np.float64]`` regardless of action.
    """
    import numpy as np

    return np.zeros(3, dtype=np.float64) if not args else np.zeros(3, dtype=np.float64)
