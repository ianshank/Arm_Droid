"""Arm controller bridging RL policy and action primitives.

Implements ``ArmControllerProtocol`` by composing a trained
``SACAgent`` for continuous actions with ``ActionPrimitives``
for discrete manipulation commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from armdroid.logging.setup import get_logger

if TYPE_CHECKING:
    from armdroid.control.primitives import ActionPrimitives
    from armdroid.control.sac_agent import SACAgent
    from armdroid.protocols import ArmEnvironmentProtocol

_log = get_logger(__name__)


class ArmController:
    """Goal-conditioned controller for arm manipulation.

    Bridges the RL policy (SACAgent) with hardware-level
    action primitives for end-to-end control.

    Args:
        agent: Trained SAC+HER policy for continuous control.
        primitives: Action primitive library for discrete commands.
    """

    def __init__(self, agent: SACAgent, primitives: ActionPrimitives) -> None:
        """Initialise arm controller.

        Args:
            agent: SAC+HER agent (may or may not be trained).
            primitives: Pre-trained grasp/place/transit primitives.
        """
        self._agent = agent
        self._primitives = primitives
        _log.info("arm_controller_init", is_trained=agent.is_trained)

    @property
    def agent(self) -> SACAgent:
        """Return the underlying SAC+HER agent (for orchestrator wiring)."""
        return self._agent

    @property
    def primitives(self) -> ActionPrimitives:
        """Return the action primitive library (for orchestrator wiring)."""
        return self._primitives

    async def execute_action(self, action: NDArray[np.float64]) -> dict[str, Any]:
        """Execute continuous action via the RL policy.

        Args:
            action: Action vector from RL policy.

        Returns:
            Execution result with keys: success, achieved_state, info.
        """
        try:
            result = await self._primitives.transit(action)
            return {
                "success": result,
                "achieved_state": action,
                "info": {},
            }
        except Exception:
            _log.error("execute_action_failed", exc_info=True)
            return {
                "success": False,
                "achieved_state": np.zeros_like(action),
                "info": {"error": "action_execution_failed"},
            }

    def build_for_env(self, env: ArmEnvironmentProtocol) -> None:
        """Bind the SAC agent to an environment if not already built.

        Args:
            env: Gymnasium-compatible environment for SAC+HER training.
        """
        if not self._agent.is_built:
            _log.info("arm_controller_building_agent")
            self._agent.build(env)

    def train_policy(self, total_timesteps: int | None = None) -> Path:
        """Train the SAC+HER policy and save a checkpoint.

        Args:
            total_timesteps: Override config total_timesteps (None = use config).

        Returns:
            Path of the saved policy checkpoint.
        """
        self._agent.train(total_timesteps)
        return self._agent.save()

    async def execute_primitive(self, primitive_name: str, target: NDArray[np.float64]) -> bool:
        """Execute pre-trained action primitive.

        Args:
            primitive_name: Primitive name ("grasp", "place", "transit", "home").
            target: Target pose for the primitive.

        Returns:
            True if primitive executed successfully.
        """
        _log.debug("execute_primitive", name=primitive_name)
        if primitive_name == "grasp":
            force = await self._primitives.grasp(target)
            return force > 0.0
        if primitive_name == "place":
            return await self._primitives.place(target)
        if primitive_name == "transit":
            return await self._primitives.transit(target)
        if primitive_name == "home":
            return await self._primitives.home()

        _log.warning("unknown_primitive", name=primitive_name)
        return False
