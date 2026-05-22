"""Vectorised environment registry (F1) - sibling of the single-env registry.

Mirrors :mod:`armdroid.environments.registry` one-for-one so the
extension-point story is identical: built-in registrations on import,
PEP 621 entry-point discovery via :data:`VEC_ENTRY_POINT_GROUP`, lazy
backend imports so the default install never reaches ``isaaclab``.

The vec registry exists alongside (not inside) the single-env one
because ``Registry[T]`` is generic over the value type; a union
factory type would lose the discriminator and force every caller into
an ``isinstance`` narrowing. See ADR-0006 for the full rationale.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

from armdroid._registry import Registry

if TYPE_CHECKING:
    from armdroid.domain.protocols import VecArmEnvironmentProtocol


#: PEP 621 entry-point group for out-of-tree vec env factories.
VEC_ENTRY_POINT_GROUP: str = "armdroid.vec_environments"

#: Subsystem kind used in :class:`Registry` log events and error messages.
VEC_REGISTRY_KIND: str = "vec_environment"


#: A vec env factory callable. Vec envs accept varied keyword arguments
#: (task config, training config, sim_isaac_cfg, ...), so the signature
#: is intentionally open with ``**Any``.
VecEnvironmentFactory: TypeAlias = "Callable[..., VecArmEnvironmentProtocol]"

_VEC_ENVIRONMENTS: Registry[VecEnvironmentFactory] = Registry(
    kind=VEC_REGISTRY_KIND,
    entry_point_group=VEC_ENTRY_POINT_GROUP,
)


def _so_arm_reach_isaac_vec_factory(
    *args: object,
    **kwargs: object,
) -> VecArmEnvironmentProtocol:
    """Lazy factory for ``SoArmReachIsaacVecEnv`` (F1).

    Mirrors the single-env ``_so_arm_reach_isaac_factory`` in
    ``registry.py`` - gates the ``isaaclab`` probe behind ``ImportError``
    so the default install never tries to load Isaac Lab. The actual
    ``SoArmReachIsaacVecEnv`` import is deferred until the factory is
    called.

    Raises:
        ArmDriverError: When the ``[isaac]`` extra is not installed.
    """
    try:
        import isaaclab.app  # noqa: F401  (probe - fails fast if [isaac] missing)
    except ImportError as exc:
        from armdroid.domain.errors import ArmDriverError

        msg = (
            f"so_arm_reach_isaac_vec env requires the [isaac] extra: {exc}. "
            'Install with `pip install -e ".[isaac]" '
            "--extra-index-url https://pypi.nvidia.com`."
        )
        raise ArmDriverError(msg) from exc

    # The factory is intentionally untyped at the *args / **kwargs boundary
    # (mirrors the single-env _so_arm_reach_isaac_factory) so registry
    # callers can pass through arbitrary keyword configs from YAML
    # overlays. The orchestration factory threads the typed configs in.
    # We runtime-validate the result satisfies VecArmEnvironmentProtocol
    # so plugin authors can't quietly return the wrong shape.
    from armdroid.domain.protocols import VecArmEnvironmentProtocol as _Proto
    from armdroid.environments.isaac.reach_vec import SoArmReachIsaacVecEnv

    env = SoArmReachIsaacVecEnv(
        *args,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )
    if not isinstance(env, _Proto):  # pragma: no cover - guard
        msg = (
            f"factory returned {type(env).__name__!r} which does not satisfy "
            "VecArmEnvironmentProtocol"
        )
        raise TypeError(msg)
    return env


_VEC_ENVIRONMENTS.register(
    "so_arm_reach_isaac_vec",
    _so_arm_reach_isaac_vec_factory,
)


def register_vec_environment(name: str, factory: VecEnvironmentFactory) -> None:
    """Register a vec env factory under ``name``."""
    _VEC_ENVIRONMENTS.register(name, factory)


def get_vec_environment(name: str) -> VecEnvironmentFactory:
    """Return the vec env factory registered under ``name``."""
    return _VEC_ENVIRONMENTS.get(name)


def available_vec_environments() -> list[str]:
    """Return the sorted list of registered vec env names."""
    return _VEC_ENVIRONMENTS.available()


def load_vec_environment_plugins() -> int:
    """Discover and register out-of-tree vec envs via entry points.

    Idempotent: ``Registry.load_entry_points`` returns ``0`` on
    subsequent calls within the same process.
    """
    return _VEC_ENVIRONMENTS.load_entry_points()


__all__ = [
    "VEC_ENTRY_POINT_GROUP",
    "VEC_REGISTRY_KIND",
    "VecEnvironmentFactory",
    "available_vec_environments",
    "get_vec_environment",
    "load_vec_environment_plugins",
    "register_vec_environment",
]
