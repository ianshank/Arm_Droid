"""Vec-env registry mirrors the single-env registry contract."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from armdroid._registry import RegistryError
from armdroid.domain.protocols import VecArmEnvironmentProtocol
from armdroid.environments.registry_vec import (
    VEC_ENTRY_POINT_GROUP,
    available_vec_environments,
    get_vec_environment,
    load_vec_environment_plugins,
    register_vec_environment,
)


def _make_factory(num_envs: int) -> Any:
    def _factory(**_kwargs: Any) -> VecArmEnvironmentProtocol:
        fake = MagicMock(spec=VecArmEnvironmentProtocol)
        fake.num_envs = num_envs
        return fake

    return _factory


def test_register_and_resolve_round_trip() -> None:
    """register + get must round-trip a factory by name."""
    register_vec_environment("test_vec_env_round_trip", _make_factory(num_envs=4))
    factory = get_vec_environment("test_vec_env_round_trip")
    resolved = factory()
    assert isinstance(resolved, VecArmEnvironmentProtocol)
    assert resolved.num_envs == 4


def test_entry_point_group_constant_pinned() -> None:
    assert VEC_ENTRY_POINT_GROUP == "armdroid.vec_environments"


def test_available_vec_environments_lists_builtin() -> None:
    """The built-in so_arm_reach_isaac_vec factory registers on import."""
    names = available_vec_environments()
    assert "so_arm_reach_isaac_vec" in names


def test_load_vec_environment_plugins_is_idempotent() -> None:
    """Repeat calls return 0 the second time per Registry.load_entry_points."""
    load_vec_environment_plugins()
    second = load_vec_environment_plugins()
    assert second == 0


def test_double_registration_rejected() -> None:
    """Different factory under same name must raise RegistryError."""
    register_vec_environment("dupe_env_vec", _make_factory(num_envs=2))
    with pytest.raises(RegistryError, match="already registered"):
        register_vec_environment("dupe_env_vec", _make_factory(num_envs=2))


def test_get_unknown_vec_environment_raises() -> None:
    """get_vec_environment surfaces RegistryError with available list."""
    with pytest.raises(RegistryError, match="unknown vec_environment"):
        get_vec_environment("does_not_exist")
