"""Top-level armdroid orchestrator composing the four arm subsystems.

The orchestrator owns the lifecycle of perception, planner, controller,
environment, and driver. It exposes three operating modes:

- :meth:`train`  — synchronous SB3 ``SAC.learn()`` loop. Wires the
  un-built ``SACAgent`` (returned from :func:`build_arm_controller`) to
  the environment via ``controller.agent.build(env)`` lazily, then calls
  ``train()`` and ``save()``.
- :meth:`rollout` — async PDDL plan + primitive execution against the
  real (or mock) driver, using perception to read symbolic state.
- :meth:`shutdown` — async driver teardown.

The async/sync split mirrors the underlying APIs: SB3 is synchronous,
the arm driver and perception are async.

PDDL plan-step args (e.g. ``["disk_1", "peg_a", "peg_c"]``) are
resolved to Cartesian XYZ targets via :func:`_resolve_target_position`
when an :class:`ArmTaskConfig` is supplied; otherwise the resolver
falls back to a zero target with a structured warning so legacy
callers that constructed the orchestrator without a task config still
work (TD-5 backward-compat).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from numpy.typing import NDArray

from armdroid.logging.setup import get_logger

if TYPE_CHECKING:
    from armdroid.config.schema.task import ArmTaskConfig
    from armdroid.domain.protocols import (
        ArmControllerProtocol,
        ArmDriverProtocol,
        ArmEnvironmentProtocol,
        ArmPerceptionProtocol,
        ArmPlannerProtocol,
        VecArmEnvironmentProtocol,
    )
    from armdroid.domain.state import PlanStep, SymbolicState

_log = get_logger(__name__)

#: Number of Cartesian dimensions for the orchestrator's primitive-target
#: vector. Sourced from the SymbolicState 3D pose convention; not a
#: hardcoded magic number — it tracks ``DetectedObject.position_m`` and
#: ``ArmTaskConfig.peg_positions`` element width.
_TARGET_DIMS: Final[int] = 3

#: Regex that recognises peg / basket names emitted by the symbolic
#: planner (lowercase ``peg_a``, ``peg_b``... and the laundry analogues).
#: The trailing capture is the index token: a single letter ``a``-``z``
#: or a digit string ``\d+``. Both are mapped to a 0-based index by
#: :func:`_index_from_token`.
_NAMED_TARGET_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<kind>peg|basket)_(?P<token>[a-z]|\d+)$",
    re.IGNORECASE,
)


class ArmOrchestrator:
    """Composes the five arm subsystems behind a stable interface.

    Args:
        perception: Perception facade (depth + detection + pose + symbolic state).
        planner: Symbolic planner (PDDL or LLM-backed).
        controller: RL controller wrapping a SACAgent + ActionPrimitives.
        environment: Gymnasium environment for SAC+HER training.
        driver: Arm hardware driver (real or mock).
    """

    def __init__(
        self,
        perception: ArmPerceptionProtocol,
        planner: ArmPlannerProtocol,
        controller: ArmControllerProtocol,
        environment: ArmEnvironmentProtocol | VecArmEnvironmentProtocol,
        driver: ArmDriverProtocol,
        task_cfg: ArmTaskConfig | None = None,
    ) -> None:
        """Initialise the orchestrator with pre-built components.

        Args:
            perception: Perception facade.
            planner: Symbolic planner.
            controller: RL controller.
            environment: Gymnasium env for training.
            driver: Arm hardware driver.
            task_cfg: Optional task config. When provided, PDDL plan-step
                args are resolved to real Cartesian XYZ targets via
                ``task_cfg.peg_positions`` / ``task_cfg.basket_positions``.
                When ``None``, the orchestrator falls back to a zero target
                vector and emits a single structured warning per rollout
                (TD-5 backward-compat path; remove the ``None`` default
                in v0.4 once all callers are migrated).
        """
        self._perception = perception
        self._planner = planner
        self._controller = controller
        self._environment = environment
        self._driver = driver
        self._task_cfg = task_cfg
        _log.info("arm_orchestrator_init", has_task_cfg=task_cfg is not None)

    @property
    def perception(self) -> ArmPerceptionProtocol:
        """Return the perception facade."""
        return self._perception

    @property
    def planner(self) -> ArmPlannerProtocol:
        """Return the symbolic planner."""
        return self._planner

    @property
    def controller(self) -> ArmControllerProtocol:
        """Return the RL controller."""
        return self._controller

    @property
    def environment(self) -> ArmEnvironmentProtocol | VecArmEnvironmentProtocol:
        """Return the Gymnasium environment (single-env or vec env)."""
        return self._environment

    @property
    def driver(self) -> ArmDriverProtocol:
        """Return the arm hardware driver."""
        return self._driver

    def train(self, total_timesteps: int | None = None) -> Path:
        """Run the SAC+HER training loop and save the resulting policy.

        Wires the SAC agent to the environment lazily on the first call
        (the factory leaves the agent un-built). Delegates to the controller's
        :meth:`build_for_env` + :meth:`train_policy` protocol methods.

        Args:
            total_timesteps: Override config ``arm_training.total_timesteps``.

        Returns:
            Filesystem path of the saved policy checkpoint.
        """
        _log.info("arm_orchestrator_building_agent")
        self._controller.build_for_env(self._environment)
        _log.info("arm_orchestrator_train_start", total_timesteps=total_timesteps)
        path = self._controller.train_policy(total_timesteps)
        _log.info("arm_orchestrator_train_complete", checkpoint=str(path))
        return path

    async def rollout(
        self,
        initial_state: SymbolicState,
        goal_state: SymbolicState,
    ) -> dict[str, Any]:
        """Plan from ``initial_state`` to ``goal_state`` and execute the plan.

        Each :class:`PlanStep` is dispatched through the controller's
        primitive layer. Failures abort and return the partial result.

        Args:
            initial_state: Starting symbolic state.
            goal_state: Target symbolic state.

        Returns:
            Dict with keys ``plan`` (list[PlanStep]), ``executed`` (int,
            number of steps successfully executed) and ``success`` (bool).
        """
        plan: list[PlanStep] = self._planner.plan(initial_state, goal_state)
        _log.info("arm_orchestrator_plan_built", n_steps=len(plan))
        executed = 0
        # Per-rollout flag so the "missing task_cfg" fallback warning fires
        # at most once even when the plan has many steps.
        warned_missing_cfg = False
        for step in plan:
            target, used_fallback = _resolve_target_position(step.args, self._task_cfg)
            if used_fallback and not warned_missing_cfg:
                _log.warning(
                    "arm_orchestrator_step_target_fallback",
                    reason=("no_task_cfg" if self._task_cfg is None else "name_unresolved"),
                    action=step.action,
                    args=step.args,
                )
                warned_missing_cfg = True
            success = await self._controller.execute_primitive(
                step.action,
                target=target,
            )
            if not success:
                _log.warning("arm_orchestrator_step_failed", step=str(step))
                break
            executed += 1
        return {
            "plan": plan,
            "executed": executed,
            "success": executed == len(plan),
        }

    async def shutdown(self) -> None:
        """Tear down the arm driver and environment."""
        _log.info("arm_orchestrator_shutdown_start")
        try:
            await self._driver.stop()
        except Exception:
            _log.exception("arm_orchestrator_driver_stop_failed")
        try:
            self._environment.close()
        except Exception:
            _log.exception("arm_orchestrator_env_close_failed")
        _log.info("arm_orchestrator_shutdown_complete")


def _index_from_token(token: str) -> int | None:
    """Map a peg/basket name suffix (e.g. ``"a"``, ``"3"``) to a 0-based index.

    Single-letter ``a``-``z`` (case-insensitive) maps to 0..25 to match
    the symbolic-planner naming convention (``peg_a``, ``peg_b``, ...).
    Decimal digits map directly to the integer they spell. Returns
    ``None`` for any other shape.

    Args:
        token: The capture group from :data:`_NAMED_TARGET_PATTERN`.

    Returns:
        0-based index, or ``None`` if the token is not recognised.
    """
    lowered = token.lower()
    if len(lowered) == 1 and lowered.isalpha():
        return ord(lowered) - ord("a")
    if lowered.isdigit():
        try:
            return int(lowered)
        except ValueError:  # pragma: no cover - isdigit guards this
            return None
    return None


def _lookup_position(
    name: str,
    task_cfg: ArmTaskConfig,
) -> NDArray[np.float64] | None:
    """Resolve a single PDDL arg name to a Cartesian XYZ target.

    Recognises ``peg_<token>`` and ``basket_<token>`` (lowercase from
    the symbolic planner; case-insensitive here for safety). Returns
    ``None`` for unrecognised names so callers can apply their own
    fallback policy.

    Args:
        name: Single PDDL argument (e.g. ``"peg_c"``).
        task_cfg: Source-of-truth for peg/basket Cartesian positions.

    Returns:
        ``NDArray`` of shape ``(_TARGET_DIMS,)`` in metres, or ``None``.
    """
    match = _NAMED_TARGET_PATTERN.match(name)
    if match is None:
        return None
    kind = match.group("kind").lower()
    index = _index_from_token(match.group("token"))
    if index is None:
        return None
    if kind == "peg":
        positions = task_cfg.peg_positions
    elif kind == "basket":
        positions = task_cfg.basket_positions
    else:  # pragma: no cover - regex restricts to peg|basket
        return None
    if not (0 <= index < len(positions)):
        return None
    return np.asarray(positions[index], dtype=np.float64)


def _step_args_to_target(args: list[str]) -> NDArray[np.float64]:
    """Backward-compat shim: zero-target adapter for legacy callers.

    Pre-TD-5, this was the orchestrator's only target resolver and always
    returned ``np.zeros(3)``. The new path is :func:`_resolve_target_position`
    which honours an optional :class:`ArmTaskConfig`. This shim delegates
    with ``task_cfg=None`` so existing imports of ``_step_args_to_target``
    (the deprecated ``armdroid.orchestrator`` module + its tests) still
    work unchanged.

    Args:
        args: PDDL action arguments — consumed for signature compatibility.

    Returns:
        ``(_TARGET_DIMS,)`` zero array, ``float64``.
    """
    target, _ = _resolve_target_position(args, task_cfg=None)
    return target


def _resolve_target_position(
    args: list[str],
    task_cfg: ArmTaskConfig | None,
) -> tuple[NDArray[np.float64], bool]:
    """Resolve PDDL plan-step args to a Cartesian XYZ target.

    Convention (from
    :class:`armdroid.planning.symbolic_planner.SymbolicPlanner`): a
    move-style action puts the destination in the LAST arg
    (``["disk_1", "peg_a", "peg_c"]`` -> ``"peg_c"``). The resolver
    walks the args **right-to-left** and returns the first
    :func:`_lookup_position` hit, so single-arg actions
    (``["peg_b"]``) and disk-only args still work without special
    casing. When nothing resolves OR ``task_cfg`` is ``None``, the
    resolver returns a zero target and signals the fallback so the
    caller can log it. Shape is fixed at :data:`_TARGET_DIMS` for
    downstream protocol stability.

    Args:
        args: PDDL action arguments.
        task_cfg: Task config providing the name->position lookup. May
            be ``None`` for backward-compat callers that constructed
            an :class:`ArmOrchestrator` without one (TD-5).

    Returns:
        Tuple ``(target, used_fallback)``. ``target`` is always a
        ``(_TARGET_DIMS,)`` ``float64`` array. ``used_fallback`` is
        ``True`` when the result is the zero fallback, signalling that
        the caller should warn (and possibly emit telemetry).
    """
    zeros = np.zeros(_TARGET_DIMS, dtype=np.float64)
    if task_cfg is None:
        return zeros, True
    for name in reversed(args):
        position = _lookup_position(name, task_cfg)
        if position is not None:
            if position.shape != (_TARGET_DIMS,):
                # Defensive: peg_positions / basket_positions are
                # validated to length-3 by ArmTaskConfig's schema, but a
                # custom task_cfg subclass could violate that. Pad/trim
                # rather than raising mid-rollout.
                truncated = position.reshape(-1)[:_TARGET_DIMS]
                fixed = np.zeros(_TARGET_DIMS, dtype=np.float64)
                fixed[: truncated.shape[0]] = truncated
                return fixed, False
            return position, False
    return zeros, True
