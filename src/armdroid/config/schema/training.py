"""RL training and curriculum configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ArmTrainingConfig(BaseModel):
    """RL training configuration for robot arm policies."""

    algorithm: Literal["sac", "ppo", "sac_her", "rsl_rl_ppo"] = Field(
        "sac_her",
        description=(
            "RL algorithm. ``sac`` / ``sac_her`` route to SACAgent (SB3); "
            "``rsl_rl_ppo`` routes to RslRlPpoAgent (Isaac Lab + RSL-RL, "
            "PR-B, requires the [isaac] extra)."
        ),
    )
    learning_rate: float = Field(3e-4, gt=0, description="Policy learning rate")
    batch_size: int = Field(256, gt=0, description="Training batch size")
    buffer_size: int = Field(1_000_000, gt=0, description="Replay buffer capacity")
    gamma: float = Field(0.99, gt=0, le=1, description="Discount factor")
    tau: float = Field(0.005, gt=0, le=1, description="Soft target update coefficient")
    total_timesteps: int = Field(1_000_000, gt=0, description="Total training timesteps")
    eval_frequency: int = Field(10_000, gt=0, description="Evaluation frequency (steps)")
    checkpoint_frequency: int = Field(50_000, gt=0, description="Checkpoint save frequency (steps)")
    n_eval_episodes: int = Field(20, gt=0, description="Episodes per evaluation")
    video_frequency: int = Field(50_000, gt=0, description="Video rollout frequency (steps)")
    her_n_sampled_goal: int = Field(4, gt=0, description="HER goal relabeling ratio")
    her_goal_selection: Literal["future", "final", "episode"] = Field(
        "future",
        description="HER goal selection strategy",
    )
    reward_grasp: float = Field(0.1, description="Reward for successful grasp")
    reward_place: float = Field(0.2, description="Reward for correct placement")
    reward_complete: float = Field(1.0, description="Reward for task completion")
    penalty_collision: float = Field(-0.5, description="Penalty for collision")
    penalty_wrong_disk: float = Field(-0.1, description="Penalty for grasping wrong disk")
    seed: int = Field(42, ge=0, description="Random seed for reproducibility")
    weights_dir: str = Field("weights/arm", description="Checkpoint output directory")
    action_delta_min: float = Field(-0.1, description="Minimum action delta per step (rad)")
    action_delta_max: float = Field(0.1, gt=0, description="Maximum action delta per step (rad)")
    distance_penalty_coeff: float = Field(
        0.01, ge=0, description="Dense distance-based reward penalty coefficient"
    )


class ArmCurriculumConfig(BaseModel):
    """Curriculum learning configuration for progressive task difficulty."""

    enabled: bool = Field(True, description="Enable curriculum learning")
    stages: list[int] = Field(
        default_factory=lambda: [1, 2, 3, 5],
        description="Curriculum stages (number of disks per stage)",
    )
    promotion_threshold: float = Field(
        0.8, gt=0, le=1, description="Success rate threshold to advance stage"
    )
    promotion_eval_episodes: int = Field(
        50, gt=0, description="Episodes to evaluate before stage promotion"
    )
    warm_start: bool = Field(True, description="Warm-start from previous stage weights")


class RslRlPpoConfig(BaseModel):
    """RSL-RL PPO hyperparameter configuration (PR-B B.3).

    Field defaults mirror MuammerBay/isaac_so_arm101's ``ReachPPORunnerCfg``
    pinned at upstream commit ``e4624dea075b00a36dbc66bebd531d191c92e8cd``
    (see ``THIRD_PARTY_NOTICES.md`` and ``ADR-0005-isaac-sim-backend.md``
    for the vendoring trail). Source:
    ``src/isaac_so_arm101/tasks/reach/agents/rsl_rl_ppo_cfg.py``.

    Consumed by ``armdroid.control.rsl_rl_agent.RslRlPpoAgent``;
    instantiation lives in ``build_arm_controller``'s explicit
    ``algorithm == "rsl_rl_ppo"`` branch (NOT the registry-dispatched
    path) so YAML overlay resolution survives — the registry's
    ``Callable[..., ArmRLAgentProtocol]`` generic cannot accommodate
    dual-config without dropping overlays via ``ArmSettings()`` re-read.
    See ADR-0005 + PR #10 review C-1.
    """

    # ------------------------------------------------------------------ #
    # Algorithm hyperparameters (PPO core)
    # ------------------------------------------------------------------ #

    num_steps_per_env: int = Field(
        default=24,
        ge=1,
        le=1024,
        description="Steps collected per env per iteration (rollout length).",
    )
    num_iterations: int = Field(
        default=1000,
        ge=1,
        description="Total PPO iterations (one rollout + update per iteration).",
    )
    num_mini_batches: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Number of minibatches the rollout is split into per epoch.",
    )
    num_learning_epochs: int = Field(
        default=8,
        ge=1,
        le=64,
        description="PPO epochs per iteration.",
    )

    # ------------------------------------------------------------------ #
    # Network
    # ------------------------------------------------------------------ #

    actor_hidden_dims: tuple[int, ...] = Field(
        default=(64, 64),
        description="Actor MLP hidden layer widths.",
    )
    critic_hidden_dims: tuple[int, ...] = Field(
        default=(64, 64),
        description="Critic MLP hidden layer widths.",
    )
    activation: Literal["elu", "relu", "tanh", "selu"] = Field(
        default="elu",
        description="Activation function for actor + critic MLPs.",
    )
    init_noise_std: float = Field(
        default=1.0,
        gt=0.0,
        description="Initial standard deviation of the action distribution.",
    )

    # ------------------------------------------------------------------ #
    # Optimisation
    # ------------------------------------------------------------------ #

    learning_rate: float = Field(
        default=1e-3,
        gt=0.0,
        description="Adam optimiser learning rate.",
    )
    schedule: Literal["adaptive", "fixed"] = Field(
        default="adaptive",
        description="LR schedule. ``adaptive`` follows desired_kl; ``fixed`` is constant.",
    )
    gamma: float = Field(
        default=0.99,
        gt=0.0,
        le=1.0,
        description="Discount factor for returns.",
    )
    gae_lambda: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description="Generalised Advantage Estimation lambda.",
    )
    clip_param: float = Field(
        default=0.2,
        gt=0.0,
        description="PPO surrogate-objective clip range epsilon.",
    )
    entropy_coef: float = Field(
        default=0.001,
        ge=0.0,
        description="Entropy regularisation coefficient.",
    )
    value_loss_coef: float = Field(
        default=1.0,
        ge=0.0,
        description="Value-function loss weighting.",
    )
    use_clipped_value_loss: bool = Field(
        default=True,
        description="Apply clipping to the value loss (matches PPO original spec).",
    )
    desired_kl: float | None = Field(
        default=0.01,
        gt=0.0,
        description="KL target for adaptive LR schedule. None disables adaptation.",
    )
    max_grad_norm: float = Field(
        default=1.0,
        gt=0.0,
        description="Gradient-norm clip for the optimiser.",
    )

    # ------------------------------------------------------------------ #
    # Logging / persistence
    # ------------------------------------------------------------------ #

    save_interval: int = Field(
        default=50,
        ge=1,
        description="Iterations between checkpoint saves.",
    )
    experiment_name: str = Field(
        default="so_arm_reach_ppo",
        description="Sub-directory name under weights_dir for this experiment.",
    )
    run_name: str = Field(
        default="",
        description="Optional run identifier (timestamped suffix when empty).",
    )
    checkpoint_path: Path | None = Field(
        default=None,
        description="Resume from this checkpoint when set; else start fresh.",
    )
    device: str = Field(
        default="cuda:0",
        min_length=1,
        description=(
            "Torch device string for the RSL-RL runner. Use 'cpu' on hosts "
            "without CUDA; 'cuda:N' to pin a specific GPU. Threaded through "
            "RslRlPpoAgent so callers do not pass it as a kwarg."
        ),
    )


__all__ = ["ArmCurriculumConfig", "ArmTrainingConfig", "RslRlPpoConfig"]
