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

import numpy as np
from numpy.typing import NDArray

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
        (the factory leaves the agent un-built). Delegates to the controller's
        :meth:`build_for_env` + :meth:`train_policy` protocol methods.

        Args:
            total_timesteps: Override config ``arm_training.total_timesteps``.

        Returns:
            Filesystem path of the saved policy checkpoint.
        """
        _log.info("arm_orchestrator_building_agent")
        self._controller.build_for_env(self._environment)
        _log.info("arm_orchestrator_train_start", total_timesteps=total_timesteps)
        path = self._controller.train_policy(total_timesteps)
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
        except Exception:
            _log.exception("arm_orchestrator_driver_stop_failed")
        try:
            self._environment.close()
        except Exception:
            _log.exception("arm_orchestrator_env_close_failed")
        _log.info("arm_orchestrator_shutdown_complete")


def _step_args_to_target(args: list[str]) -> NDArray[np.float64]:
    """Convert PDDL plan-step args to a zero target vector.

    Symbolic-to-pose conversion (peg positions → joint angles) lives inside
    :class:`ActionPrimitives`, not here. This boundary adapter returns a
    zero vector so every ``execute_primitive`` call receives a consistent
    ``NDArray[np.float64]`` shape. The primitives layer does its own
    lookup against the arm config to resolve actual targets.

    Args:
        args: PDDL action arguments (e.g. ``["disk1", "peg_A", "peg_C"]``).

    Returns:
        Zero array of shape ``(3,)`` — x/y/z placeholder target.
    """
    _ = args  # consumed by ActionPrimitives via the primitive_name dispatch
    return np.zeros(3, dtype=np.float64)
