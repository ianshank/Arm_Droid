"""Vectorised SoArmReach environment (F1) - VecArmEnvironmentProtocol impl.

Owns Isaac Lab's ``ManagerBasedRLEnv`` directly with ``num_envs >= 1``;
no ``_TensorAdapter`` (the vec protocol IS torch-native). Coordinates
with :mod:`armdroid.hardware.isaac_sim._app_state` so the Kit singleton
is not double-booted when both the driver and the vec env are
constructed in the same process.

The factory function :func:`_build_isaac_env` is monkeypatched in unit
tests so the production code path is exercised without an
``isaaclab`` install.

Coverage-omit: this module is in ``[tool.coverage.run].omit`` via the
existing ``*/armdroid/environments/isaac/*`` wildcard. Tests live under
``tests/unit/environments/`` (pure-Python, monkeypatched) and
``tests/isaac/`` (gated on ``ARMDROID_ISAAC_RUN=1`` + a CUDA GPU).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from armdroid.logging.setup import get_logger
from armdroid.telemetry import (
    SPAN_ENV_VEC_CLOSE,
    SPAN_ENV_VEC_KIT_BOOT,
    SPAN_ENV_VEC_RESET,
    SPAN_ENV_VEC_STEP,
    get_telemetry,
)

if TYPE_CHECKING:
    import torch

    from armdroid.config.schema.sim_isaac import ArmSimIsaacConfig
    from armdroid.config.schema.task import ArmTaskConfig
    from armdroid.config.schema.training import ArmTrainingConfig


_log = get_logger(__name__)


def _build_isaac_env(
    task_cfg: ArmTaskConfig,
    training_cfg: ArmTrainingConfig,
    sim_isaac_cfg: ArmSimIsaacConfig,
) -> Any:
    """Construct the underlying ``ManagerBasedRLEnv``.

    Isolated so unit tests can monkeypatch it without importing
    ``isaaclab``. Mirrors the single-env ``SoArmReachIsaacEnv._ensure_built``
    body - same ``gym.make`` arguments, the only observable diff is
    ``num_envs >= 1`` flowing through unchanged.
    """
    import gymnasium as gym

    from armdroid.environments.isaac.tasks import reach as _  # noqa: F401

    return gym.make(
        sim_isaac_cfg.reach_env_id,
        num_envs=sim_isaac_cfg.num_envs,
        disable_env_checker=sim_isaac_cfg.disable_env_checker,
    )


class SoArmReachIsaacVecEnv:
    """Vectorised reach env conforming to ``VecArmEnvironmentProtocol``.

    Permits ``num_envs >= 1``. The orchestration factory still routes
    ``num_envs == 1`` to the single-env path by default; this class
    exists so callers may adopt the vec protocol unconditionally.

    Args:
        task_cfg: Manipulation task config.
        training_cfg: RL training config.
        sim_isaac_cfg: Optional Isaac Sim config; falls back to a
            fresh default via
            :func:`armdroid.config.schema.sim_isaac._default_sim_cfg`
            (matches the single-env constructor).
    """

    def __init__(
        self,
        task_cfg: ArmTaskConfig,
        training_cfg: ArmTrainingConfig,
        *,
        sim_isaac_cfg: ArmSimIsaacConfig | None = None,
    ) -> None:
        if sim_isaac_cfg is None:
            from armdroid.config.schema.sim_isaac import _default_sim_cfg

            sim_isaac_cfg = _default_sim_cfg()

        self._task_cfg = task_cfg
        self._training_cfg = training_cfg
        self._sim_cfg = sim_isaac_cfg
        self._isaac_env: Any = None
        self._kit_booted_here: bool = False
        _log.info(
            "so_arm_reach_isaac_vec_init",
            num_envs=self._sim_cfg.num_envs,
            headless=self._sim_cfg.headless,
            env_id=self._sim_cfg.reach_env_id,
        )

    def _ensure_built(self) -> None:
        """Lazy build the underlying gym env on first use.

        Coordinates with :mod:`armdroid.hardware.isaac_sim._app_state`
        so the Kit singleton is observed (probe) before construction
        and marked launched after a successful boot. This prevents a
        later :class:`IsaacSimDriver` construction in the same process
        from double-booting Kit (and vice versa).
        """
        if self._isaac_env is not None:
            return

        from armdroid.hardware.isaac_sim import _app_state

        already_launched = _app_state.is_app_launched()
        with get_telemetry().start_span(
            SPAN_ENV_VEC_KIT_BOOT,
            num_envs=self._sim_cfg.num_envs,
            kit_already_launched=already_launched,
        ):
            self._isaac_env = _build_isaac_env(
                self._task_cfg, self._training_cfg, self._sim_cfg,
            )
            if not already_launched:
                _app_state.mark_launched()
                self._kit_booted_here = True
            _log.info(
                "so_arm_reach_isaac_vec_built",
                env_id=self._sim_cfg.reach_env_id,
                kit_booted_here=self._kit_booted_here,
            )

    @property
    def num_envs(self) -> int:
        """Number of parallel envs (mirrors ``ArmSimIsaacConfig.num_envs``)."""
        return self._sim_cfg.num_envs

    def reset(
        self, *, seed: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Reset all parallel envs."""
        with get_telemetry().start_span(
            SPAN_ENV_VEC_RESET, num_envs=self.num_envs, seed=seed,
        ):
            self._ensure_built()
            return self._isaac_env.reset(seed=seed)

    def step(
        self, action: torch.Tensor,
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Step all parallel envs."""
        with get_telemetry().start_span(
            SPAN_ENV_VEC_STEP, num_envs=self.num_envs,
        ):
            self._ensure_built()
            return self._isaac_env.step(action)

    def close(self) -> None:
        """Release all parallel envs."""
        with get_telemetry().start_span(SPAN_ENV_VEC_CLOSE):
            _log.info("so_arm_reach_isaac_vec_close")
            if self._isaac_env is not None:
                self._isaac_env.close()
                self._isaac_env = None

    def as_runner_env(self) -> Any:
        """Return the underlying ``ManagerBasedRLEnv`` for RL runners.

        Lazy-builds on first call. Replacement for the legacy
        ``env._isaac_env`` reach-through.
        """
        self._ensure_built()
        return self._isaac_env


__all__ = ["SoArmReachIsaacVecEnv", "_build_isaac_env"]
