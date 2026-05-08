"""Smoke tests for per-subsystem registries (drivers, planners, envs, etc.)."""

from __future__ import annotations

import pytest

from armdroid._registry import RegistryError


def test_driver_registry_built_ins() -> None:
    from armdroid.environments.tower_of_hanoi import TowerOfHanoiEnv  # noqa: F401
    from armdroid.hardware.esp32_json_driver import Esp32JsonDriver
    from armdroid.hardware.mock_arm_driver import MockArmDriver
    from armdroid.hardware.registry import (
        available_drivers,
        get_driver,
        register_driver,
    )

    names = available_drivers()
    assert "mock" in names
    assert "esp32" in names
    assert get_driver("mock") is MockArmDriver
    assert get_driver("esp32") is Esp32JsonDriver

    # Re-registration with same factory must be idempotent.
    register_driver("mock", MockArmDriver)


def test_driver_registry_rejects_collision() -> None:
    from armdroid.hardware.mock_arm_driver import MockArmDriver
    from armdroid.hardware.registry import register_driver

    class _OtherDriver:
        pass

    with pytest.raises(RegistryError):
        register_driver("mock", _OtherDriver)  # type: ignore[arg-type]
    # Restore canonical mapping for downstream tests.
    register_driver("mock", MockArmDriver)


def test_planner_registry_built_ins() -> None:
    from armdroid.planning.registry import available_planners, get_planner
    from armdroid.planning.symbolic_planner import SymbolicPlanner

    assert "pyperplan" in available_planners()
    assert get_planner("pyperplan") is SymbolicPlanner


def test_environment_registry_built_ins() -> None:
    from armdroid.environments.laundry_sorting import LaundrySortingEnv
    from armdroid.environments.registry import (
        available_environments,
        get_environment,
    )
    from armdroid.environments.tower_of_hanoi import TowerOfHanoiEnv

    names = available_environments()
    assert "tower_of_hanoi" in names
    assert "laundry_sorting" in names
    assert get_environment("tower_of_hanoi") is TowerOfHanoiEnv
    assert get_environment("laundry_sorting") is LaundrySortingEnv


def test_perception_registry_built_ins() -> None:
    from armdroid.perception.facade import ArmPerception
    from armdroid.perception.registry import (
        available_perception_backends,
        get_perception_backend,
    )

    assert "default" in available_perception_backends()
    assert get_perception_backend("default") is ArmPerception


def test_rl_agent_registry_built_ins() -> None:
    from armdroid.control.registry import available_rl_agents, get_rl_agent
    from armdroid.control.sac_agent import SACAgent

    assert "sac" in available_rl_agents()
    assert get_rl_agent("sac") is SACAgent


def test_registry_unknown_lookup_raises() -> None:
    from armdroid.hardware.registry import get_driver

    with pytest.raises(RegistryError, match="unknown driver"):
        get_driver("nonexistent_driver")
