"""F1 vec env public-API regression guards.

Pins the importable surface of the vectorised env path so accidental
renames / removals during future refactors trip CI immediately. Mirrors
the single-env coverage in :mod:`tests.regression.test_baseline` and
:mod:`tests.regression.test_legacy_compat`.

Scope (the canonical F1 public surface that v0.2.x consumers may rely on):

* ``armdroid.domain.protocols.VecArmEnvironmentProtocol`` -
  ``@runtime_checkable``, declares ``num_envs`` + ``reset`` + ``step``
  + ``close`` + ``as_runner_env``.
* ``armdroid.domain.protocols.VecArmRLAgentProtocol`` -
  ``@runtime_checkable``, declares ``build_vec`` + ``train_vec`` +
  ``predict`` + ``save`` + ``load`` + ``is_built`` + ``is_trained``.
* ``armdroid.environments.registry_vec`` exports
  ``register_vec_environment`` / ``get_vec_environment`` /
  ``available_vec_environments`` / ``load_vec_environment_plugins``
  and the constants ``VEC_ENTRY_POINT_GROUP`` /
  ``VEC_REGISTRY_KIND`` / ``VecEnvironmentFactory``.
* Built-in registration for ``so_arm_reach_isaac_vec`` is present on
  import (no entry-point installation required).
* ``armdroid.telemetry`` exports the six F1 SPAN constants with the
  documented string values.
* ``armdroid.orchestration.factory._VEC_TASK_REGISTRY_NAMES`` maps
  ``so_arm_reach_isaac -> so_arm_reach_isaac_vec``.
* ``armdroid.config.schema.training.RslRlPpoConfig.device`` is a
  string field.
* ``armdroid.config.schema.sim_isaac.ArmSimIsaacConfig.disable_env_checker``
  is a bool field defaulting to ``True``.

Marked ``regression`` so CI can filter into a dedicated stage.
"""

from __future__ import annotations

from typing import Protocol, get_type_hints, runtime_checkable

import pytest

pytestmark = pytest.mark.regression


class TestVecProtocolSurface:
    """Importable public surface for the F1 vec protocols."""

    def test_vec_arm_environment_protocol_importable(self) -> None:
        from armdroid.domain.protocols import VecArmEnvironmentProtocol

        assert issubclass(VecArmEnvironmentProtocol, Protocol)

    def test_vec_arm_environment_protocol_is_runtime_checkable(self) -> None:
        from armdroid.domain.protocols import VecArmEnvironmentProtocol

        # The runtime_checkable decorator sets _is_runtime_protocol.
        assert getattr(
            VecArmEnvironmentProtocol,
            "_is_runtime_protocol",
            False,
        )

    def test_vec_arm_environment_protocol_required_members(self) -> None:
        from armdroid.domain.protocols import VecArmEnvironmentProtocol

        required = {"num_envs", "reset", "step", "close", "as_runner_env"}
        assert required.issubset(set(dir(VecArmEnvironmentProtocol)))

    def test_vec_arm_rl_agent_protocol_importable_and_runtime_checkable(
        self,
    ) -> None:
        from armdroid.domain.protocols import VecArmRLAgentProtocol

        assert issubclass(VecArmRLAgentProtocol, Protocol)
        assert getattr(
            VecArmRLAgentProtocol,
            "_is_runtime_protocol",
            False,
        )

    def test_vec_arm_rl_agent_protocol_required_members(self) -> None:
        from armdroid.domain.protocols import VecArmRLAgentProtocol

        required = {
            "build_vec",
            "train_vec",
            "predict",
            "save",
            "load",
            "is_built",
            "is_trained",
        }
        assert required.issubset(set(dir(VecArmRLAgentProtocol)))

    def test_both_protocols_in_protocols_dunder_all(self) -> None:
        from armdroid.domain import protocols as _p

        assert "VecArmEnvironmentProtocol" in _p.__all__
        assert "VecArmRLAgentProtocol" in _p.__all__


class TestVecRegistrySurface:
    """Public helpers + constants on armdroid.environments.registry_vec."""

    def test_registry_vec_module_exports(self) -> None:
        from armdroid.environments import registry_vec

        for name in (
            "VEC_ENTRY_POINT_GROUP",
            "VEC_REGISTRY_KIND",
            "VecEnvironmentFactory",
            "available_vec_environments",
            "get_vec_environment",
            "load_vec_environment_plugins",
            "register_vec_environment",
        ):
            assert name in registry_vec.__all__, f"missing {name} in __all__"
            assert hasattr(registry_vec, name)

    def test_entry_point_group_constant_pinned(self) -> None:
        from armdroid.environments.registry_vec import VEC_ENTRY_POINT_GROUP

        assert VEC_ENTRY_POINT_GROUP == "armdroid.vec_environments"

    def test_registry_kind_constant_pinned(self) -> None:
        from armdroid.environments.registry_vec import VEC_REGISTRY_KIND

        assert VEC_REGISTRY_KIND == "vec_environment"

    def test_so_arm_reach_isaac_vec_registered_on_import(self) -> None:
        from armdroid.environments.registry_vec import available_vec_environments

        assert "so_arm_reach_isaac_vec" in available_vec_environments()


class TestVecTelemetrySpans:
    """SPAN_* constants on armdroid.telemetry for the vec path."""

    def test_vec_spans_pinned(self) -> None:
        from armdroid import telemetry

        expected = {
            "SPAN_AGENT_BUILD_VEC": "armdroid.agent.build_vec",
            "SPAN_AGENT_TRAIN_VEC": "armdroid.agent.train_vec",
            "SPAN_ENV_VEC_RESET": "armdroid.env.vec_reset",
            "SPAN_ENV_VEC_STEP": "armdroid.env.vec_step",
            "SPAN_ENV_VEC_CLOSE": "armdroid.env.vec_close",
            "SPAN_ENV_VEC_KIT_BOOT": "armdroid.env.vec_kit_boot",
        }
        for name, value in expected.items():
            assert hasattr(telemetry, name)
            assert getattr(telemetry, name) == value
            assert name in telemetry.__all__


class TestVecFactoryDispatch:
    """Factory mapping + dispatch helpers stay importable and stable."""

    def test_vec_capable_algorithms_contains_rsl_rl_ppo(self) -> None:
        from armdroid.orchestration.factory import _VEC_CAPABLE_ALGORITHMS

        assert "rsl_rl_ppo" in _VEC_CAPABLE_ALGORITHMS

    def test_vec_task_registry_names_mapping(self) -> None:
        from armdroid.orchestration.factory import _VEC_TASK_REGISTRY_NAMES

        assert _VEC_TASK_REGISTRY_NAMES["so_arm_reach_isaac"] == ("so_arm_reach_isaac_vec")

    def test_should_use_vec_helper_importable(self) -> None:
        from armdroid.orchestration.factory import _should_use_vec

        assert callable(_should_use_vec)


class TestVecConfigFields:
    """F1-introduced config fields are present with documented defaults."""

    def test_rsl_rl_ppo_config_has_device_field(self) -> None:
        from armdroid.config.schema.training import RslRlPpoConfig

        cfg = RslRlPpoConfig()
        assert isinstance(cfg.device, str)
        # Field is overridable via YAML; default is the CUDA pin.
        assert cfg.device.startswith(("cuda", "cpu"))

    def test_arm_sim_isaac_config_has_disable_env_checker_field(self) -> None:
        from armdroid.config.schema.sim_isaac import ArmSimIsaacConfig

        cfg = ArmSimIsaacConfig()
        # Default is True (Isaac Lab's recommendation).
        assert cfg.disable_env_checker is True
        # Overridable: round-trip via construction.
        override = ArmSimIsaacConfig(disable_env_checker=False)
        assert override.disable_env_checker is False


class TestVecControllerDispatch:
    """ArmControllerProtocol.build_for_env accepts the union type."""

    def test_arm_controller_protocol_build_for_env_accepts_union(self) -> None:
        from armdroid.domain.protocols import ArmControllerProtocol

        # get_type_hints resolves forward refs even with from __future__
        # import annotations enabled in the module. The hint must mention
        # both single-env and vec env protocol names.
        hints = get_type_hints(ArmControllerProtocol.build_for_env)
        env_hint = hints.get("env")
        assert env_hint is not None
        # Union shapes report __args__; runtime_checkable Protocols are
        # included so we just assert both names appear in the string repr.
        env_hint_str = repr(env_hint)
        assert "ArmEnvironmentProtocol" in env_hint_str
        assert "VecArmEnvironmentProtocol" in env_hint_str


@runtime_checkable
class _MinimalVecEnv(Protocol):
    """Local mirror used to confirm structural compatibility cheaply."""

    @property
    def num_envs(self) -> int: ...

    def reset(self) -> tuple[dict[str, object], dict[str, object]]: ...

    def step(
        self,
        action: object,
    ) -> tuple[
        dict[str, object],
        object,
        object,
        object,
        dict[str, object],
    ]: ...

    def close(self) -> None: ...

    def as_runner_env(self) -> object: ...


class TestVecAgentConcreteSurface:
    """Concrete RslRlPpoAgent class shape - guards against accidental rename.

    The protocol surface tests above pin the abstract contract; this
    suite pins the concrete implementation so a rename of
    ``build_vec`` / ``train_vec`` / ``_iterations_for`` on
    :class:`armdroid.control.rsl_rl_agent.RslRlPpoAgent` fails CI
    even when the protocol stays satisfied (per reviewer finding M-4).
    """

    def test_rsl_rl_ppo_agent_has_build_vec_method(self) -> None:
        from armdroid.control.rsl_rl_agent import RslRlPpoAgent

        assert hasattr(RslRlPpoAgent, "build_vec")
        assert callable(RslRlPpoAgent.build_vec)

    def test_rsl_rl_ppo_agent_has_train_vec_method(self) -> None:
        from armdroid.control.rsl_rl_agent import RslRlPpoAgent

        assert hasattr(RslRlPpoAgent, "train_vec")
        assert callable(RslRlPpoAgent.train_vec)

    def test_rsl_rl_ppo_agent_has_iterations_for_helper(self) -> None:
        """Shared helper between ``train`` and ``train_vec``."""
        from armdroid.control.rsl_rl_agent import RslRlPpoAgent

        assert hasattr(RslRlPpoAgent, "_iterations_for")
        assert callable(RslRlPpoAgent._iterations_for)

    def test_rsl_rl_ppo_agent_instance_satisfies_both_protocols(self) -> None:
        """Concrete agent structurally satisfies single-env AND vec protocols."""
        from armdroid.config.schema.training import (
            ArmTrainingConfig,
            RslRlPpoConfig,
        )
        from armdroid.control.rsl_rl_agent import RslRlPpoAgent
        from armdroid.domain.protocols import (
            ArmRLAgentProtocol,
            VecArmRLAgentProtocol,
        )

        agent = RslRlPpoAgent(
            ppo_cfg=RslRlPpoConfig(),
            training_cfg=ArmTrainingConfig(algorithm="rsl_rl_ppo"),
            device="cpu",
        )
        assert isinstance(agent, ArmRLAgentProtocol)
        assert isinstance(agent, VecArmRLAgentProtocol)


class TestVecEnvConcreteSurface:
    """Concrete SoArmReachIsaacVecEnv class shape - guards against rename."""

    def test_so_arm_reach_isaac_vec_env_has_as_runner_env_method(self) -> None:
        from armdroid.environments.isaac.reach_vec import SoArmReachIsaacVecEnv

        assert hasattr(SoArmReachIsaacVecEnv, "as_runner_env")
        assert callable(SoArmReachIsaacVecEnv.as_runner_env)

    def test_so_arm_reach_isaac_vec_env_has_num_envs_property(self) -> None:
        from armdroid.environments.isaac.reach_vec import SoArmReachIsaacVecEnv

        # num_envs is declared as a @property on the concrete class.
        assert isinstance(
            SoArmReachIsaacVecEnv.__dict__.get("num_envs"),
            property,
        )


class TestVecProtocolStructuralConformance:
    """Mocks and lightweight stubs satisfying the protocol cleanly."""

    def test_minimal_stub_satisfies_protocol(self) -> None:
        from armdroid.domain.protocols import VecArmEnvironmentProtocol

        class _Stub:
            num_envs = 4

            def reset(self) -> tuple[dict[str, object], dict[str, object]]:
                return {}, {}

            def step(
                self,
                action: object,
            ) -> tuple[
                dict[str, object],
                object,
                object,
                object,
                dict[str, object],
            ]:
                return {}, 0.0, False, False, {}

            def close(self) -> None: ...

            def as_runner_env(self) -> object:
                return self

        assert isinstance(_Stub(), VecArmEnvironmentProtocol)
