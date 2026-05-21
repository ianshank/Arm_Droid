"""Arm controller bridging RL policy and action primitives.

Implements ``ArmControllerProtocol`` by composing any RL agent that
satisfies ``ArmRLAgentProtocol`` (SAC+HER today; PR-B's RSL-RL PPO next)
with ``ActionPrimitives`` for discrete manipulation commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from armdroid.domain.protocols import (
    ArmEnvironmentProtocol,
    ArmRLAgentProtocol,
    VecArmEnvironmentProtocol,
    VecArmRLAgentProtocol,
)
from armdroid.logging.setup import get_logger

if TYPE_CHECKING:
    from armdroid.control.primitives import ActionPrimitives

_log = get_logger(__name__)


class ArmController:
    """Goal-conditioned controller for arm manipulation.

    Bridges any RL policy that satisfies ``ArmRLAgentProtocol`` with
    hardware-level action primitives for end-to-end control.

    Args:
        agent: RL policy conforming to :class:`ArmRLAgentProtocol`.
        primitives: Action primitive library for discrete commands.
    """

    def __init__(
        self,
        agent: ArmRLAgentProtocol,
        primitives: ActionPrimitives,
    ) -> None:
        """Initialise arm controller.

        Args:
            agent: RL agent (may or may not be trained).
            primitives: Pre-trained grasp/place/transit primitives.
        """
        self._agent = agent
        self._primitives = primitives
        self._is_vec_path: bool = False
        _log.info("arm_controller_init", is_trained=agent.is_trained)

    @property
    def agent(self) -> ArmRLAgentProtocol:
        """Return the underlying RL agent (for orchestrator wiring)."""
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

    def build_for_env(
        self,
        env: ArmEnvironmentProtocol | VecArmEnvironmentProtocol,
    ) -> None:
        """Bind the underlying RL agent to an environment if not already built.

        Dispatches between the single-env :meth:`ArmRLAgentProtocol.build`
        and the vec :meth:`VecArmRLAgentProtocol.build_vec` based on the
        runtime protocol type of ``env`` (F1). For backwards-compat,
        agents that do not implement ``build_vec`` fall back to the
        single-env path.

        Args:
            env: Single-env or vec-env conforming to the relevant
                protocol. Vec envs are detected via
                :class:`VecArmEnvironmentProtocol`'s ``num_envs``
                attribute.
        """
        if self._agent.is_built:
            return
        if isinstance(env, VecArmEnvironmentProtocol):
            # Vec env REQUIRES a vec-capable agent; falling back to
            # agent.build() would silently feed torch tensors into a
            # single-env policy expecting numpy scalars. Refuse the
            # mismatch explicitly. (Gemini Code Assist review fix.)
            if not isinstance(self._agent, VecArmRLAgentProtocol):
                msg = (
                    f"agent {type(self._agent).__name__!r} does not "
                    "implement VecArmRLAgentProtocol but env is "
                    "vectorised; choose a vec-capable algorithm "
                    "(e.g. rsl_rl_ppo) or set arm_sim_isaac.num_envs == 1."
                )
                raise ValueError(msg)
            _log.info(
                "arm_controller_building_agent_vec",
                num_envs=env.num_envs,
            )
            self._agent.build_vec(env)
            self._is_vec_path = True
            return
        _log.info("arm_controller_building_agent")
        self._agent.build(env)

    def train_policy(self, total_timesteps: int | None = None) -> Path:
        """Train the policy and save a checkpoint.

        Dispatches to :meth:`VecArmRLAgentProtocol.train_vec` if the
        agent was previously bound via the vec path
        (:meth:`build_for_env` with a :class:`VecArmEnvironmentProtocol`).
        Otherwise falls back to the single-env :meth:`train`.

        Args:
            total_timesteps: Override config total_timesteps (None = use config).

        Returns:
            Path of the saved policy checkpoint.
        """
        # _is_vec_path flips True only in build_for_env after a successful
        # build_vec call, which itself guards on
        # isinstance(self._agent, VecArmRLAgentProtocol). The isinstance
        # check here is belt-and-braces - it also lets mypy narrow the
        # agent type so the train_vec call type-checks cleanly.
        # (Gemini Code Assist review fix.)
        if self._is_vec_path and isinstance(self._agent, VecArmRLAgentProtocol):
            self._agent.train_vec(total_timesteps)
        else:
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
