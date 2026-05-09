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

            raw_env = getattr(env, "_isaac_env", env)
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

        RSL-RL's RunnerCfg is a dataclass-like object; we keep the cfg
        construction local to this method so the import stays inside
        ``build()`` (not at module top).
        """
        # Lazy import — RSL-RL only available with [isaac] extra.
        # NB: we import the exact OnPolicyRunner config class lazily;
        # the field set comes from RslRlPpoConfig (B.3) which mirrors
        # MuammerBay's ReachPPORunnerCfg.
        ppo = self._ppo_cfg
        # Construct via plain dict + attribute setting since RSL-RL's
        # cfg dataclass is upstream (lazy-imported above the type cast).
        return {
            "seed": self._training_cfg.seed,
            "device": self._device,
            "num_steps_per_env": ppo.num_steps_per_env,
            "num_learning_epochs": ppo.num_learning_epochs,
            "num_mini_batches": ppo.num_mini_batches,
            "learning_rate": ppo.learning_rate,
            "schedule": ppo.schedule,
            "gamma": ppo.gamma,
            "lam": ppo.gae_lambda,
            "entropy_coef": ppo.entropy_coef,
            "value_loss_coef": ppo.value_loss_coef,
            "use_clipped_value_loss": ppo.use_clipped_value_loss,
            "clip_param": ppo.clip_param,
            "max_grad_norm": ppo.max_grad_norm,
            "desired_kl": ppo.desired_kl,
            "save_interval": ppo.save_interval,
            "experiment_name": ppo.experiment_name,
            "run_name": ppo.run_name,
            "actor_hidden_dims": list(ppo.actor_hidden_dims),
            "critic_hidden_dims": list(ppo.critic_hidden_dims),
            "activation": ppo.activation,
            "init_noise_std": ppo.init_noise_std,
        }


__all__ = ["RslRlPpoAgent"]
