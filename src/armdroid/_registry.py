"""Generic plugin registry used by armdroid subsystems.

This module defines a small, dependency-free :class:`Registry` that backs
the per-subsystem plugin lookup tables introduced in v0.2 (drivers,
planners, perception backends, environments, RL agents).

Design goals:
- **Zero magic**: built-in implementations register themselves explicitly
  on import of the corresponding ``armdroid.<subsystem>.registry`` module.
  No metaclasses, no decorators that touch class creation order.
- **Out-of-tree extension via entry points**: third-party packages can
  declare entry points in groups such as ``armdroid.drivers`` and have
  their factories discovered via :meth:`Registry.load_entry_points`.
- **Strict typing**: the registry is generic over the value type so that
  ``get`` returns a precise factory type instead of ``Any``.

The registry stores *factories* (zero-arg-or-callable producers), not
already-constructed instances, so that callers can build instances with
their own configuration objects.

See ``docs/architecture/PHASES.md`` and
``docs/architecture/ADR/ADR-0001-enterprise-layering.md`` for the
rationale; ADR-0002 will document the registry contract once Phase 2b
lands the factory dispatch rewrite.
"""

from __future__ import annotations

from importlib import metadata
from typing import Generic, TypeVar

from armdroid.logging.setup import get_logger

T = TypeVar("T")

_log = get_logger(__name__)


class RegistryError(Exception):
    """Raised when a registry lookup or registration fails."""


class Registry(Generic[T]):
    """A typed name → factory registry with optional entry-point discovery.

    Args:
        kind: Human-readable subsystem name used in log events and error
            messages (e.g. ``"driver"``, ``"planner"``).
        entry_point_group: Optional :pep:`621` entry-point group name
            (e.g. ``"armdroid.drivers"``). When set,
            :meth:`load_entry_points` discovers and registers any
            third-party plugins declared under that group.

    Notes:
        Registration is idempotent: re-registering the same name with the
        same factory is a no-op. Re-registering with a *different*
        factory raises :class:`RegistryError` to surface plugin
        collisions early. Use :meth:`override` for tests that
        intentionally swap an implementation.
    """

    def __init__(self, kind: str, entry_point_group: str | None = None) -> None:
        """Initialise the registry."""
        self._kind = kind
        self._entry_point_group = entry_point_group
        self._items: dict[str, T] = {}
        self._entry_points_loaded = False

    @property
    def kind(self) -> str:
        """Return the subsystem name (``"driver"``, ``"planner"``, ...)."""
        return self._kind

    @property
    def entry_point_group(self) -> str | None:
        """Return the configured entry-point group, if any."""
        return self._entry_point_group

    def register(self, name: str, factory: T) -> None:
        """Register ``factory`` under ``name``.

        Args:
            name: Lookup key (lowercase, hyphen-separated by convention).
            factory: The callable / class object to associate with the
                key.

        Raises:
            RegistryError: If a *different* factory is already registered
                for ``name``.
        """
        existing = self._items.get(name)
        if existing is not None and existing is not factory:
            raise RegistryError(
                f"{self._kind} '{name}' is already registered to "
                f"{existing!r}; refusing to overwrite with {factory!r}. "
                f"Use Registry.override() if intentional."
            )
        self._items[name] = factory
        _log.debug("registry.registered", kind=self._kind, name=name)

    def override(self, name: str, factory: T) -> None:
        """Forcefully replace the factory at ``name`` (for tests/plugins)."""
        self._items[name] = factory
        _log.warning("registry.overridden", kind=self._kind, name=name)

    def unregister(self, name: str) -> None:
        """Remove ``name`` from the registry. No-op if not present."""
        self._items.pop(name, None)

    def get(self, name: str) -> T:
        """Return the factory registered under ``name``.

        Args:
            name: Lookup key.

        Raises:
            RegistryError: If ``name`` is not registered. The error
                message lists currently-available names.
        """
        try:
            return self._items[name]
        except KeyError:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise RegistryError(
                f"unknown {self._kind} '{name}'. Registered: {available}."
            ) from None

    def available(self) -> list[str]:
        """Return the sorted list of registered names."""
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        """Membership test against registered names."""
        return isinstance(name, str) and name in self._items

    def load_entry_points(self) -> int:
        """Discover and register plugins from the configured entry-point group.

        Safe to call multiple times — only the first call performs the
        scan. Returns the number of plugins newly registered. If no
        entry-point group is configured, returns ``0``.

        Failures inside individual entry-point ``load()`` calls are
        logged at error level but do not abort discovery — a single
        broken plugin must not block the rest of the system.
        """
        if self._entry_points_loaded or self._entry_point_group is None:
            return 0
        self._entry_points_loaded = True
        loaded = 0
        eps = metadata.entry_points(group=self._entry_point_group)
        for ep in eps:
            try:
                factory = ep.load()
            except Exception as exc:  # report and continue
                _log.error(
                    "registry.entry_point_failed",
                    kind=self._kind,
                    group=self._entry_point_group,
                    name=ep.name,
                    error=str(exc),
                )
                continue
            try:
                self.register(ep.name, factory)
                loaded += 1
            except RegistryError as exc:
                _log.error(
                    "registry.entry_point_collision",
                    kind=self._kind,
                    name=ep.name,
                    error=str(exc),
                )
        if loaded:
            _log.info(
                "registry.entry_points_loaded",
                kind=self._kind,
                group=self._entry_point_group,
                count=loaded,
            )
        return loaded
