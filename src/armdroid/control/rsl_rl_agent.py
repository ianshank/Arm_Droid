"""RslRlPpoAgent — RSL-RL PPO as ArmRLAgentProtocol (PR-B B.13).

Lazy-imports ``rsl_rl`` inside ``build()`` so the module is importable
on default installs (the ``[isaac]`` extra is required only at runtime).
Mirrors the ``SACAgent`` pattern for telemetry spans + structlog events.

Key design points:
* Constructor takes BOTH ``ppo_cfg`` and ``training_cfg`` — the
  registry's ``Callable[..., ArmRLAgentProtocol]`` factory cannot
  accommodate dual-config without dropping YAML overlays via
  ``ArmSettings()`` re-read (peer-review C-1). The orchestrator's
  ``build_arm_controller`` branches on
  ``algorithm == "rsl_rl_ppo"`` to thread both configs through.
* ``build(env)`` reaches through to ``env._isaac_env`` — RSL-RL's
  ``OnPolicyRunner`` requires the raw ``ManagerBasedRLEnv`` (torch
  tensor IO), not the protocol-wrapped numpy single-env view exposed
  by ``SoArmReachIsaacEnv``.
* ``predict(obs)`` converts numpy obs dict → torch tensors on
  ``self._device``, runs inference, converts back to numpy.

Coverage-omit: this module is in ``[tool.coverage.run].omit``. Tests
live under ``tests/unit/control/`` (gated on
``pytest.importorskip("rsl_rl")``) and ``tests/isaac/``
(gated on ``ARMDROID_ISAAC_RUN=1``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from armdroid.logging.setup import get_logger
from armdroid.telemetry import (
    SPAN_AGENT_BUILD,
    SPAN_AGENT_LOAD,
    SPAN_AGENT_PREDICT,
    SPAN_AGENT_SAVE,
    SPAN_AGENT_TRAIN,
    get_telemetry,
)

if TYPE_CHECKING:
    from armdroid.config.schema.training import ArmTrainingConfig, RslRlPpoConfig
    from armdroid.domain.protocols import ArmEnvironmentProtocol

_log = get_logger(__name__)


class RslRlPpoAgent:
    """RSL-RL PPO agent satisfying ``ArmRLAgentProtocol``.

    Args:
        ppo_cfg: RSL-RL hyperparameters (B.3).
        training_cfg: Generic RL training config — total_timesteps,
            weights_dir, seed.
        device: Torch device for the runner (default ``"cuda:0"``).
            Tests pass ``"cpu"`` to skip GPU requirements.
    """

    def __init__(
        self,
        ppo_cfg: RslRlPpoConfig,
        training_cfg: ArmTrainingConfig,
        *,
        device: str = "cuda:0",
    ) -> None:
        self._ppo_cfg = ppo_cfg
        self._training_cfg = training_cfg
        self._device = device
        self._runner: Any = None
        self._is_trained = False
        _log.info(
            "rsl_rl_ppo_agent_init",
            device=device,
            num_iterations=ppo_cfg.num_iterations,
        )

    # ------------------------------------------------------------------ #
    # ArmRLAgentProtocol surface
    # ------------------------------------------------------------------ #

    def build(self, env: ArmEnvironmentProtocol) -> None:
        """Build the OnPolicyRunner.

        RSL-RL's runner requires a torch-tensor vec-env (Isaac Lab's
        ``ManagerBasedRLEnv``), NOT the protocol-wrapped numpy
        single-env view. ``SoArmReachIsaacEnv`` exposes the raw env
        via ``_isaac_env``; we reach through here.
        """
        with get_telemetry().start_span(SPAN_AGENT_BUILD, device=self._device):
            try:
                from rsl_rl.runners import OnPolicyRunner
            except ImportError as exc:
                _log.error("rsl_rl_ppo_build_failed", error=str(exc))
                msg = (
                    f"RSL-RL not installed: {exc}. "
                    'Install with `pip install -e ".[isaac]" '
                    "--extra-index-url https://pypi.nvidia.com`."
                )
                raise ImportError(msg) from exc

            # SoArmReachIsaacEnv lazy-builds _isaac_env on first reset/step;
            # without forcing the build here, OnPolicyRunner would receive
            # None and fail at runtime when build() is called before reset().
            ensure_built = getattr(env, "_ensure_built", None)
            if callable(ensure_built):
                ensure_built()
            raw_env = getattr(env, "_isaac_env", None) or env
            if raw_env is None:
                msg = (
                    "env._isaac_env is None even after _ensure_built(); "
                    "cannot construct OnPolicyRunner."
                )
                raise RuntimeError(msg)
            runner_cfg = self._build_runner_cfg()
            self._runner = OnPolicyRunner(
                raw_env,
                runner_cfg,
                log_dir=None,
                device=self._device,
            )
            _log.info(
                "rsl_rl_ppo_agent_built",
                experiment_name=self._ppo_cfg.experiment_name,
            )

    def train(self, total_timesteps: int | None = None) -> None:
        """Run the RSL-RL training loop.

        Args:
            total_timesteps: Override config total. ``None`` →
                ``ppo_cfg.num_iterations`` iterations from the
                upstream config (RSL-RL counts iterations, not
                timesteps).
        """
        if self._runner is None:
            msg = "Runner not built. Call build(env) first."
            raise RuntimeError(msg)

        iterations = (
            int(total_timesteps // self._ppo_cfg.num_steps_per_env)
            if total_timesteps is not None
            else self._ppo_cfg.num_iterations
        )

        with get_telemetry().start_span(
            SPAN_AGENT_TRAIN,
            total_timesteps=total_timesteps,
            num_iterations=iterations,
        ):
            _log.info(
                "rsl_rl_ppo_training_start",
                total_timesteps=total_timesteps,
                num_iterations=iterations,
            )
            self._runner.learn(num_learning_iterations=iterations)
            self._is_trained = True
            _log.info(
                "rsl_rl_ppo_training_complete",
                total_timesteps_completed=iterations * self._ppo_cfg.num_steps_per_env,
            )

    def predict(
        self,
        observation: dict[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Predict greedy action for ``observation``.

        Converts numpy obs dict → torch on device, runs the inference
        policy, converts back to numpy float64.
        """
        if self._runner is None:
            msg = "Runner not built. Call build(env) first."
            raise RuntimeError(msg)

        with get_telemetry().start_span(SPAN_AGENT_PREDICT):
            try:
                import torch

                obs_t = {k: torch.from_numpy(v).to(self._device) for k, v in observation.items()}
                policy = self._runner.get_inference_policy(device=self._device)
                with torch.no_grad():
                    action_t = policy(obs_t)
                action_np = action_t.cpu().numpy().astype(np.float64)
                # RSL-RL's policy returns shape (num_envs, action_dim);
                # ArmRLAgentProtocol.predict() (matching SACAgent / the
                # ActionPrimitives path) expects a flat (action_dim,)
                # vector. Squeeze the leading env dim when num_envs == 1
                # — the only configuration the protocol path supports —
                # so callers receive the documented shape.
                if action_np.ndim >= 2 and action_np.shape[0] == 1:
                    action_np = action_np.reshape(action_np.shape[1:])
                _log.debug("rsl_rl_ppo_predict", obs_keys=list(observation.keys()))
                return action_np
            except Exception as exc:
                _log.error(
                    "rsl_rl_ppo_predict_failed",
                    error=str(exc),
                    exc_info=True,
                )
                raise

    def save(self, path: str | None = None) -> Path:
        """Save model checkpoint."""
        if self._runner is None:
            msg = "Runner not built."
            raise RuntimeError(msg)
        save_path = (
            Path(path)
            if path is not None
            else Path(self._training_cfg.weights_dir) / f"{self._ppo_cfg.experiment_name}.pt"
        )
        with get_telemetry().start_span(SPAN_AGENT_SAVE, path=str(save_path)):
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self._runner.save(str(save_path))
            _log.info("rsl_rl_ppo_model_saved", path=str(save_path))
        return save_path

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        if self._runner is None:
            msg = "Runner not built. Call build(env) before load(path)."
            raise RuntimeError(msg)
        with get_telemetry().start_span(SPAN_AGENT_LOAD, path=path):
            self._runner.load(path)
            self._is_trained = True
            _log.info("rsl_rl_ppo_model_loaded", path=path)

    @property
    def is_trained(self) -> bool:
        """Whether ``train()`` has completed at least once."""
        return self._is_trained

    @property
    def is_built(self) -> bool:
        """Whether ``build()`` has bound a runner."""
        return self._runner is not None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_runner_cfg(self) -> Any:
        """Build the RSL-RL ``OnPolicyRunnerCfg`` from PPO + training fields.

        RSL-RL >= 2.0 (the version Isaac Lab 2.3 ships with) requires a
        typed config object with nested ``policy`` and ``algorithm``
        attribute groups — a flat dict raises ``AttributeError`` on the
        first access inside ``OnPolicyRunner.__init__``.

        The shape mirrors the vendored ``ReachPPORunnerCfg`` at
        ``armdroid.environments.isaac.tasks.reach.agents.rsl_rl_ppo_cfg``;
        every numeric/string knob is sourced from :class:`RslRlPpoConfig`
        (no hardcoded defaults here).

        PR-11 review fix C3 (gemini #3): the previous flat-dict return
        would have crashed at runtime on the first ``build()`` call.
        """
        from isaaclab_rl.rsl_rl import (
            RslRlOnPolicyRunnerCfg,
            RslRlPpoActorCriticCfg,
            RslRlPpoAlgorithmCfg,
        )

        ppo = self._ppo_cfg

        policy = RslRlPpoActorCriticCfg(
            init_noise_std=ppo.init_noise_std,
            actor_hidden_dims=list(ppo.actor_hidden_dims),
            critic_hidden_dims=list(ppo.critic_hidden_dims),
            activation=ppo.activation,
        )
        algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=ppo.value_loss_coef,
            use_clipped_value_loss=ppo.use_clipped_value_loss,
            clip_param=ppo.clip_param,
            entropy_coef=ppo.entropy_coef,
            num_learning_epochs=ppo.num_learning_epochs,
            num_mini_batches=ppo.num_mini_batches,
            learning_rate=ppo.learning_rate,
            schedule=ppo.schedule,
            gamma=ppo.gamma,
            lam=ppo.gae_lambda,
            desired_kl=ppo.desired_kl,
            max_grad_norm=ppo.max_grad_norm,
        )

        # RslRlOnPolicyRunnerCfg is a configclass (Isaac Lab @configclass);
        # construct via no-arg + attribute assignment, which is the
        # idiomatic upstream pattern (see ReachPPORunnerCfg).
        runner_cfg = RslRlOnPolicyRunnerCfg()
        runner_cfg.seed = self._training_cfg.seed
        runner_cfg.device = self._device
        runner_cfg.num_steps_per_env = ppo.num_steps_per_env
        runner_cfg.max_iterations = ppo.num_iterations
        runner_cfg.save_interval = ppo.save_interval
        runner_cfg.experiment_name = ppo.experiment_name
        runner_cfg.run_name = ppo.run_name
        runner_cfg.resume = False
        runner_cfg.empirical_normalization = False
        runner_cfg.policy = policy
        runner_cfg.algorithm = algorithm
        return runner_cfg


__all__ = ["RslRlPpoAgent"]
