"""Value objects exchanged across armdroid layer boundaries.

These types are immutable (or close to it) and depend only on standard
library + ``numpy``. They are part of the stable public surface re-exported
from :mod:`armdroid`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ArmState:
    """Snapshot of arm telemetry returned by ``ArmDriverProtocol.read_state``.

    Attributes:
        joint_positions: Joint angles in radians (gripper joint, when present,
            is normalised to ``[0, 1]`` with ``0`` = open, ``1`` = closed).
        joint_velocities: Joint angular velocities in rad/s. Open-loop and
            mock drivers may report zeros when no motion is in progress.
        is_moving: ``True`` if any joint is currently slewing toward a
            commanded target.
        estop_active: ``True`` if the firmware (or mock) is currently latched
            in emergency-stop and rejecting motion commands.
        timestamp_s: Monotonic timestamp (seconds) when the state was sampled.
    """

    joint_positions: tuple[float, ...]
    joint_velocities: tuple[float, ...]
    is_moving: bool
    estop_active: bool
    timestamp_s: float


@dataclass(frozen=True, slots=True)
class ArmAction:
    """Action emitted by a controller or vision-language agent.

    Stored as a tuple of joints (not an ``NDArray``) so frozen-dataclass
    equality and hashing remain meaningful and match the existing
    :class:`ArmState` convention. Use :meth:`from_array` at the boundary
    when a backend produces a numpy array (e.g. Gemini VLA's joint
    deltas).

    Attributes:
        joint_targets: Target joint values. Units follow the driver's
            convention: radians for rotational joints, normalised
            ``[0, 1]`` for the gripper joint when present.
        gripper: Convenience scalar for the gripper command in
            ``[0, 1]``; ``0`` = open, ``1`` = closed. May duplicate the
            gripper joint inside ``joint_targets`` when the driver
            exposes it as a joint.
        timestamp_s: Optional monotonic timestamp (seconds) for when the
            action was generated. ``None`` if untimed.
    """

    joint_targets: tuple[float, ...]
    gripper: float = 0.0
    timestamp_s: float | None = None

    @classmethod
    def from_array(
        cls,
        joint_targets: NDArray[np.float64],
        gripper: float = 0.0,
        timestamp_s: float | None = None,
    ) -> ArmAction:
        """Build an :class:`ArmAction` from an ``NDArray`` of joint targets.

        Args:
            joint_targets: Numpy array of joint targets, shape ``(dof,)``.
            gripper: Gripper command in ``[0, 1]``.
            timestamp_s: Optional monotonic timestamp.

        Returns:
            Immutable :class:`ArmAction` whose ``joint_targets`` field
            holds a tuple copy of the array values.
        """
        return cls(
            joint_targets=tuple(float(x) for x in np.asarray(joint_targets).ravel()),
            gripper=float(gripper),
            timestamp_s=timestamp_s,
        )


@dataclass(frozen=True, slots=True)
class Verdict:
    """Decision returned by a :class:`SafetyGuardProtocol` check.

    Attributes:
        allowed: ``True`` if the guard approves the plan or action.
        reason: Free-form explanation when ``allowed`` is ``False`` (or
            an informational note when ``True``). Empty by default so
            the permissive default does not require a justification.
        guard_name: Name of the guard that produced the verdict; useful
            when chained guards are composed via Swiss-cheese semantics.
    """

    allowed: bool
    reason: str = ""
    guard_name: str = ""


@dataclass(frozen=True, slots=True)
class InteractionEvent:
    """Single event surfaced by an :class:`InteractionSessionProtocol`.

    Captures the shape of Live-API events that the orchestrator routes
    into the replanner mid-rollout. Domain-agnostic so any future
    interaction backend (Gemini Live, OpenAI Realtime) can re-emit
    events through this contract.

    Attributes:
        kind: Event type, one of ``"audio"``, ``"text"``, ``"frame"``,
            ``"replan_request"``, or ``"session_end"``.
        text: Optional decoded text payload (e.g. ASR transcript).
        timestamp_s: Optional monotonic timestamp.
    """

    kind: str
    text: str = ""
    timestamp_s: float | None = None


@dataclass(frozen=True, slots=True)
class SceneInsight:
    """Output of an agentic-vision scene reasoner (e.g. Gemini ER 1.6).

    Captures the model's suggested image transformations and free-form
    notes so the caller can act on them without re-asking the model.

    Attributes:
        crops: Suggested crops as ``(x_min, y_min, x_max, y_max)`` pixel
            rectangles. Empty when no crop is recommended.
        rotations_deg: Suggested rotations in degrees, paired with the
            same index in :attr:`crops` when both are non-empty.
        notes: Free-form natural-language commentary from the model.
    """

    crops: tuple[tuple[int, int, int, int], ...] = ()
    rotations_deg: tuple[float, ...] = ()
    notes: str = ""


class DetectedObject:
    """Detected object with pose, classification, and optional semantics.

    Attributes:
        object_id: Unique identifier for the detected object.
        class_name: Object class (e.g., "disk_1", "shirt_white").
        confidence: Detection confidence score [0, 1].
        position_m: 3D position in world frame (x, y, z) metres.
        orientation_rad: Orientation as Euler angles (roll, pitch, yaw) radians.
        bbox: 2D bounding box [x_min, y_min, x_max, y_max] pixels.
        affordances: Tuple of affordance tags emitted by open-vocabulary
            detectors (e.g. ``("graspable", "stackable")``). Empty for
            legacy closed-vocabulary detectors.
        is_fragile: Open-vocabulary detector hint that the object is
            fragile. ``False`` for legacy detectors.
        is_fixed: Open-vocabulary detector hint that the object is
            bolted-down / immovable. ``False`` for legacy detectors.
        semantic_tags: Free-form tags emitted by a VLM detector
            (e.g. ``("metal", "tool")``). Empty for legacy detectors.
        text_query: When the detection was triggered by a natural-language
            query, this records the originating prompt for traceability.
            ``None`` for class-driven detections.
    """

    __slots__ = (
        "affordances",
        "bbox",
        "class_name",
        "confidence",
        "is_fixed",
        "is_fragile",
        "object_id",
        "orientation_rad",
        "position_m",
        "semantic_tags",
        "text_query",
    )

    def __init__(
        self,
        object_id: str,
        class_name: str,
        confidence: float,
        position_m: NDArray[np.float64],
        orientation_rad: NDArray[np.float64],
        bbox: NDArray[np.float64],
        *,
        affordances: tuple[str, ...] = (),
        is_fragile: bool = False,
        is_fixed: bool = False,
        semantic_tags: tuple[str, ...] = (),
        text_query: str | None = None,
    ) -> None:
        """Initialise detected object.

        Positional parameters preserve the v0.2 ctor signature so every
        existing call site keeps working. Open-vocabulary detector
        outputs (affordances, fragility, semantic tags, originating
        text query) are keyword-only with falsy defaults.

        Args:
            object_id: Unique object identifier.
            class_name: Object class name.
            confidence: Detection confidence [0, 1].
            position_m: World-frame position (3,).
            orientation_rad: Euler angles (3,).
            bbox: Bounding box (4,).
            affordances: Optional affordance tags from a VLM detector.
            is_fragile: Optional fragility hint from a VLM detector.
            is_fixed: Optional immovable-object hint from a VLM detector.
            semantic_tags: Optional free-form tags from a VLM detector.
            text_query: Optional originating natural-language prompt.
        """
        self.object_id = object_id
        self.class_name = class_name
        self.confidence = confidence
        self.position_m = position_m
        self.orientation_rad = orientation_rad
        self.bbox = bbox
        self.affordances = affordances
        self.is_fragile = is_fragile
        self.is_fixed = is_fixed
        self.semantic_tags = semantic_tags
        self.text_query = text_query


class SymbolicState:
    """Symbolic state representation for PDDL planning.

    Attributes:
        predicates: Set of active PDDL predicates (e.g., ``on(disk1, peg_A)``).
        objects: Mapping of object names to their types.
    """

    __slots__ = ("objects", "predicates")

    def __init__(
        self,
        predicates: frozenset[str],
        objects: dict[str, str],
    ) -> None:
        """Initialise symbolic state.

        Args:
            predicates: Active PDDL predicates.
            objects: Object name -> type mapping.
        """
        self.predicates = predicates
        self.objects = objects

    def __eq__(self, other: object) -> bool:
        """Check equality by predicates and objects."""
        if not isinstance(other, SymbolicState):
            return NotImplemented
        return self.predicates == other.predicates and self.objects == other.objects

    def __hash__(self) -> int:
        """Hash by predicates."""
        return hash(self.predicates)


class PlanStep:
    """Single step in a symbolic plan.

    Attributes:
        action: Action name (e.g., "move").
        args: Action arguments (e.g., ["disk1", "peg_A", "peg_C"]).
    """

    __slots__ = ("action", "args")

    def __init__(self, action: str, args: list[str]) -> None:
        """Initialise plan step.

        Args:
            action: PDDL action name.
            args: Action argument list.
        """
        self.action = action
        self.args = args

    def __repr__(self) -> str:
        """Readable representation."""
        return f"{self.action}({', '.join(self.args)})"


__all__ = [
    "ArmAction",
    "ArmState",
    "DetectedObject",
    "InteractionEvent",
    "PlanStep",
    "SceneInsight",
    "SymbolicState",
    "Verdict",
]
