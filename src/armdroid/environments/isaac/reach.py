"""SoArmReachIsaacEnv — ArmEnvironmentProtocol-conformant Isaac Lab wrapper.

PR-B B.11a. Wraps Isaac Lab 2.3 ``ManagerBasedRLEnv`` (registered as
``Isaac-SO-ARM100-Reach-v0`` by the vendored ``tasks/reach/__init__.py``)
in a ``_TensorAdapter`` (``num_envs == 1`` only) so it satisfies
``ArmEnvironmentProtocol``.

For vectorised training (``num_envs > 1``), bypass this class and use
the raw ``_isaac_env`` attribute via Isaac Lab's runner directly. This
is documented + tested in PR-B B.13's RslRlPpoAgent which reaches
through to ``env._isaac_env`` for the vectorised path.

Coverage-omit: this module is in ``[tool.coverage.run].omit``. Tests
live under ``tests/isaac/`` and only run with ``ARMDROID_ISAAC_RUN=1``
+ a CUDA GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from armdroid.environments._tensor_adapter import _TensorAdapter
from armdroid.logging.setup import get_logger
from armdroid.telemetry import (
    SPAN_ENV_CLOSE,
    SPAN_ENV_RENDER,
    SPAN_ENV_RESET,
    SPAN_ENV_STEP,
    get_telemetry,
)

if TYPE_CHECKING:
    from armdroid.config.schema.sim_isaac import ArmSimIsaacConfig
    from armdroid.config.schema.task import ArmTaskConfig
    from armdroid.config.schema.training import ArmTrainingConfig

_log = get_logger(__name__)


class SoArmReachIsaacEnv:
    """Isaac Lab reach env wrapper conforming to ``ArmEnvironmentProtocol``.

    The protocol path requires ``num_envs == 1``. Vectorised training
    bypasses this wrapper via the ``_isaac_env`` attribute (see B.13's
    ``RslRlPpoAgent.build`` which reaches through).

    Args:
        task_cfg: Manipulation task config (max_episode_steps, etc).
        training_cfg: RL training config (action delta bounds, etc).
        dof: Robot DoF; default 6 for SO-ARM101 (5 arm + 1 gripper).
        sim_isaac_cfg: Optional Isaac Sim config; defaults to
            :func:`armdroid.config.schema.sim_isaac._default_sim_cfg`.
    """

    def __init__(
        self,
        task_cfg: ArmTaskConfig,
        training_cfg: ArmTrainingConfig,
        *,
        dof: int = 6,
        sim_isaac_cfg: ArmSimIsaacConfig | None = None,
    ) -> None:
        if sim_isaac_cfg is None:
            from armdroid.config.schema.sim_isaac import _default_sim_cfg

            sim_isaac_cfg = _default_sim_cfg()

        self._task_cfg = task_cfg
        self._training_cfg = training_cfg
        self._dof = dof
        self._sim_cfg = sim_isaac_cfg
        self._adapter = _TensorAdapter(num_envs=self._sim_cfg.num_envs)
        self._isaac_env: Any = None  # populated by _ensure_built()
        _log.info(
            "so_arm_reach_isaac_init",
            num_envs=self._sim_cfg.num_envs,
            headless=self._sim_cfg.headless,
            env_id=self._sim_cfg.reach_env_id,
            dof=self._dof,
        )

    def _ensure_built(self) -> None:
        """Lazy build the underlying gym env on first reset / step.

        Triggers gym.register via the vendored
        ``armdroid.environments.isaac.tasks.reach`` __init__, then
        ``gym.make`` boots the ``ManagerBasedRLEnv``.
        """
        if self._isaac_env is not None:
            return
        # Lazy gym + tasks/reach import — both require isaaclab.
        import gymnasium as gym

        from armdroid.environments.isaac.tasks import reach as _  # noqa: F401

        self._isaac_env = gym.make(
            self._sim_cfg.reach_env_id,
            num_envs=self._sim_cfg.num_envs,
            disable_env_checker=True,
        )
        _log.info("so_arm_reach_isaac_built", env_id=self._sim_cfg.reach_env_id)

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset env to initial state."""
        with get_telemetry().start_span(SPAN_ENV_RESET, seed=seed):
            self._ensure_built()
            _log.debug("so_arm_reach_isaac_reset", seed=seed)
            return self._adapter.reset_to_protocol(self._isaac_env.reset(seed=seed))

    def step(
        self,
        action: NDArray[np.float64],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute one environment step."""
        with get_telemetry().start_span(SPAN_ENV_STEP, dof=self._dof):
            isaac_action = self._adapter.action_from_protocol(action)
            return self._adapter.step_to_protocol(self._isaac_env.step(isaac_action))

    def render(self) -> NDArray[np.uint8] | None:
        """Render current state.

        Returns None when running headless (the default).
        """
        with get_telemetry().start_span(SPAN_ENV_RENDER):
            if self._isaac_env is None or self._sim_cfg.headless:
                return None
            frame = self._isaac_env.render()
            return None if frame is None else np.asarray(frame, dtype=np.uint8)

    def close(self) -> None:
        """Clean up environment resources."""
        with get_telemetry().start_span(SPAN_ENV_CLOSE):
            _log.info("so_arm_reach_isaac_close")
            if self._isaac_env is not None:
                self._isaac_env.close()
                self._isaac_env = None


__all__ = ["SoArmReachIsaacEnv"]
