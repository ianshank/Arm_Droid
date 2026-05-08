"""armdroid CLI entry point.

Subcommands:
- ``train``  : Run the SAC+HER training loop.
- ``rollout``: Plan + execute a Tower of Hanoi solve against the driver.
- ``sim``    : Step the environment for ``--episodes`` episodes (no training).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

from armdroid.config.schema import ArmSettings, load_settings
from armdroid.factory import build_arm_orchestrator
from armdroid.logging.setup import configure_logging, get_logger
from armdroid.protocols import SymbolicState


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the argparse Namespace.

    Args:
        argv: Optional argv override (for tests).

    Returns:
        Parsed args.
    """
    parser = argparse.ArgumentParser(
        prog="armdroid",
        description="Robot arm manipulation platform — Tower of Hanoi + laundry sorting.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=[],
        help=(
            "YAML config overlay file. Repeat to layer multiple overlays "
            "(merged in order on top of config/default.yaml)."
        ),
    )
    parser.add_argument(
        "--mock-hardware",
        action="store_true",
        help="Force mock hardware mode (no SO-ARM100 USB required).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Run SAC+HER training loop.")
    train_p.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override config arm_training.total_timesteps.",
    )

    rollout_p = sub.add_parser("rollout", help="Plan and execute a Hanoi solve.")
    rollout_p.add_argument(
        "--num-disks",
        type=int,
        default=None,
        help="Override config arm_task.num_disks.",
    )

    sim_p = sub.add_parser("sim", help="Step the env for --episodes (no training).")
    sim_p.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of sim episodes to run.",
    )

    return parser.parse_args(argv)


def cli_entry() -> None:  # pragma: no cover
    """Main CLI entry point — dispatches to the chosen subcommand."""
    args = _parse_args()
    cfg = load_settings(*args.config)
    if args.mock_hardware:
        cfg = cfg.model_copy(update={"mock_hardware": True})

    configure_logging(cfg.logging)
    log = get_logger(__name__)
    log.info(
        "armdroid_starting",
        command=args.command,
        mock=cfg.mock_hardware,
        config_files=[str(p) for p in args.config],
    )

    if args.command == "train":
        _train(cfg, total_timesteps=args.total_timesteps)
    elif args.command == "rollout":
        asyncio.run(_rollout(cfg, num_disks=args.num_disks))
    elif args.command == "sim":
        _sim(cfg, episodes=args.episodes)
    else:
        log.error("armdroid_unknown_command", command=args.command)
        sys.exit(2)


def _train(cfg: ArmSettings, total_timesteps: int | None = None) -> None:  # pragma: no cover
    """Build the orchestrator and run the SAC+HER training loop.

    Args:
        cfg: Root settings.
        total_timesteps: Override config training timesteps.
    """
    orch = build_arm_orchestrator(cfg)
    log = get_logger(__name__)
    try:
        path = orch.train(total_timesteps=total_timesteps)
        log.info("armdroid_train_complete", checkpoint=str(path))
    finally:
        asyncio.run(orch.shutdown())


async def _rollout(cfg: ArmSettings, num_disks: int | None = None) -> None:  # pragma: no cover
    """Plan + execute a Tower of Hanoi solve against the driver.

    Args:
        cfg: Root settings.
        num_disks: Optional override for the disk count.
    """
    if num_disks is not None:
        cfg = cfg.model_copy(deep=True)
        cfg.arm_task.num_disks = num_disks
    orch = build_arm_orchestrator(cfg)
    log = get_logger(__name__)
    try:
        # Trivial state pair — the planner generates from config alone for Hanoi.
        empty = SymbolicState(predicates=frozenset(), objects={})
        result = await orch.rollout(empty, empty)
        log.info(
            "armdroid_rollout_complete",
            n_steps=len(result["plan"]),
            executed=result["executed"],
            success=result["success"],
        )
    finally:
        await orch.shutdown()


def _sim(cfg: ArmSettings, episodes: int) -> None:  # pragma: no cover
    """Step the environment for ``episodes`` rollouts without training.

    Args:
        cfg: Root settings.
        episodes: Number of episodes to step.
    """
    orch = build_arm_orchestrator(cfg)
    log = get_logger(__name__)
    env = orch.environment
    total_reward = 0.0
    try:
        for ep in range(episodes):
            _obs, _info = env.reset(seed=ep)
            ep_reward = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = np.zeros(cfg.arm.dof, dtype=np.float64)
                _obs, reward, terminated, truncated, _info = env.step(action)
                ep_reward += float(reward)
            total_reward += ep_reward
            log.info("armdroid_sim_episode", episode=ep, reward=ep_reward)
        log.info("armdroid_sim_complete", episodes=episodes, mean_reward=total_reward / episodes)
    finally:
        asyncio.run(orch.shutdown())


if __name__ == "__main__":  # pragma: no cover
    cli_entry()
