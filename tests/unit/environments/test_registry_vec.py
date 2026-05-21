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


def test_so_arm_reach_isaac_vec_factory_success_path_returns_vec_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the isaaclab probe succeeds, the factory constructs the vec env.

    Covers the post-ImportError body of ``_so_arm_reach_isaac_vec_factory``:
    the lazy class import, the constructor call, and the runtime
    ``isinstance`` check on the result. Uses a fake ``isaaclab.app`` module
    so the probe passes, and monkeypatches ``_build_isaac_env`` so the
    underlying ``gym.make`` is bypassed.
    """
    import sys
    import types

    from armdroid.environments.isaac import reach_vec

    # Inject a stub isaaclab.app so the probe import succeeds.
    fake_isaaclab = types.ModuleType("isaaclab")
    fake_app = types.ModuleType("isaaclab.app")
    fake_isaaclab.app = fake_app  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_app)

    # Bypass the real gym.make inside _build_isaac_env.
    from tests.helpers.fake_isaac_env import make_fake_isaac_env

    monkeypatch.setattr(
        reach_vec,
        "_build_isaac_env",
        lambda *_a, **_kw: make_fake_isaac_env(num_envs=2),
    )

    from armdroid.config.schema.sim_isaac import ArmSimIsaacConfig
    from armdroid.config.schema.task import ArmTaskConfig
    from armdroid.config.schema.training import ArmTrainingConfig

    factory = get_vec_environment("so_arm_reach_isaac_vec")
    env = factory(
        task_cfg=ArmTaskConfig(),
        training_cfg=ArmTrainingConfig(),
        sim_isaac_cfg=ArmSimIsaacConfig(num_envs=2),
    )
    assert isinstance(env, VecArmEnvironmentProtocol)
    assert env.num_envs == 2


def test_so_arm_reach_isaac_vec_factory_raises_armdrivererror_without_isaaclab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lazy isaaclab probe must raise ArmDriverError with an install hint.

    Forces the ``import isaaclab.app`` line inside
    ``_so_arm_reach_isaac_vec_factory`` to raise ImportError by removing
    the ``isaaclab`` name from sys.modules and shadowing the finder.
    Mirrors the single-env registry's coverage of the same path.
    """
    import builtins
    import sys

    from armdroid.domain.errors import ArmDriverError

    monkeypatch.delitem(sys.modules, "isaaclab", raising=False)
    monkeypatch.delitem(sys.modules, "isaaclab.app", raising=False)

    real_import = builtins.__import__

    def _import_fail_on_isaaclab(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "isaaclab.app" or name.startswith("isaaclab."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_fail_on_isaaclab)

    factory = get_vec_environment("so_arm_reach_isaac_vec")
    with pytest.raises(ArmDriverError, match=r"requires the \[isaac\] extra"):
        factory()
