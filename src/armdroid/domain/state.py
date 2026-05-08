"""Value objects exchanged across armdroid layer boundaries.

These types are immutable (or close to it) and depend only on standard
library + ``numpy``. They are part of the stable public surface re-exported
from :mod:`armdroid`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


class DetectedObject:
    """Detected object with pose and classification.

    Attributes:
        object_id: Unique identifier for the detected object.
        class_name: Object class (e.g., "disk_1", "shirt_white").
        confidence: Detection confidence score [0, 1].
        position_m: 3D position in world frame (x, y, z) metres.
        orientation_rad: Orientation as Euler angles (roll, pitch, yaw) radians.
        bbox: 2D bounding box [x_min, y_min, x_max, y_max] pixels.
    """

    __slots__ = ("bbox", "class_name", "confidence", "object_id", "orientation_rad", "position_m")

    def __init__(
        self,
        object_id: str,
        class_name: str,
        confidence: float,
        position_m: NDArray[np.float64],
        orientation_rad: NDArray[np.float64],
        bbox: NDArray[np.float64],
    ) -> None:
        """Initialise detected object.

        Args:
            object_id: Unique object identifier.
            class_name: Object class name.
            confidence: Detection confidence [0, 1].
            position_m: World-frame position (3,).
            orientation_rad: Euler angles (3,).
            bbox: Bounding box (4,).
        """
        self.object_id = object_id
        self.class_name = class_name
        self.confidence = confidence
        self.position_m = position_m
        self.orientation_rad = orientation_rad
        self.bbox = bbox


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


__all__ = ["ArmState", "DetectedObject", "PlanStep", "SymbolicState"]
