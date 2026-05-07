"""Tests for armdroid.main argparse wiring."""

from __future__ import annotations

import pytest

from armdroid.main import _parse_args


class TestArgParsing:
    """argparse subcommand wiring."""

    def test_train_subcommand(self) -> None:
        args = _parse_args(["train"])
        assert args.command == "train"
        assert args.total_timesteps is None

    def test_train_with_timesteps(self) -> None:
        args = _parse_args(["train", "--total-timesteps", "1000"])
        assert args.total_timesteps == 1000

    def test_rollout_subcommand(self) -> None:
        args = _parse_args(["rollout"])
        assert args.command == "rollout"

    def test_rollout_with_disks(self) -> None:
        args = _parse_args(["rollout", "--num-disks", "5"])
        assert args.num_disks == 5

    def test_sim_subcommand(self) -> None:
        args = _parse_args(["sim"])
        assert args.command == "sim"
        assert args.episodes == 5

    def test_sim_with_episodes(self) -> None:
        args = _parse_args(["sim", "--episodes", "10"])
        assert args.episodes == 10

    def test_config_path_accumulates(self) -> None:
        args = _parse_args(["--config", "a.yaml", "--config", "b.yaml", "train"])
        assert len(args.config) == 2

    def test_mock_hardware_flag(self) -> None:
        args = _parse_args(["--mock-hardware", "sim"])
        assert args.mock_hardware is True

    def test_missing_subcommand_raises(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args([])
