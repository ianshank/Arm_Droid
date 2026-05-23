# ADR-0008: Open-vocabulary detector discriminator (Phase A)

## Status

Proposed (landing in Phase A of branch
`claude/gemini-robotics-embodied-reasoning-KrlEw`; concrete backend in
Phase B).

## Context

Gemini ER 1.6 introduces open-vocabulary object detection: developers
phrase queries in natural language ("the tool that looks the most
used") and the model returns grounded bounding boxes plus rich
metadata (affordances, fragility, semantic tags). The current
armdroid perception stack is closed-vocabulary: `ObjectDetector`
loads a YOLO weights file and returns a fixed-vocabulary
`list[DetectedObject]` whose fields cover position, orientation, and
bounding box only.

Two design pressures shape the discriminator:

1. **`runtime_checkable Protocol` is method-name strict.** Any method
   we add to `ObjectDetectorProtocol` becomes a hard structural
   requirement on every existing detector. There is no "optional
   method" semantic. So either we add the new methods to every
   detector concretely, or we live with structurally divergent
   detectors.
2. **`DetectedObject` is a `__slots__` class with positional ctor.**
   The widening must preserve the positional six-arg ctor signature
   so every existing call site in `src/armdroid/perception/` and
   `tests/` keeps working without modification.

The peer review of plan v1 surfaced both pressures (HIGH H2 and
BLOCKER B4).

## Decision

1. **Detector kind discriminator.** Add
   `ArmPerceptionConfig.detector_kind: Literal["yolo", "gemini_er"]`
   defaulting to `"yolo"`. The orchestration factory consults this
   value when building the perception facade (Phase B).
2. **Detector registry.** Phase B introduces
   `armdroid.perception.detector_registry` mirroring the shape of
   `armdroid.hardware.registry`. Built-in keys: `"yolo"` (eager,
   in-tree YOLO) and `"gemini_er"` (lazy import behind the `[gemini]`
   extra).
3. **`ObjectDetectorProtocol` widening (concrete, not optional).**
   Add `detect_text(rgb, prompt) -> list[DetectedObject]` and
   `infer_affordances(detections, rgb) -> list[DetectedObject]` to
   the protocol. The existing YOLO `ObjectDetector` implements both
   concretely: `detect_text` returns `[]`, `infer_affordances`
   returns the detections unchanged. This preserves
   `isinstance(YoloDetector, ObjectDetectorProtocol)` and avoids the
   "optional method" anti-pattern.
4. **`DetectedObject` widening via keyword-only params.** Phase A
   extends the `__slots__` tuple and the `__init__` body with five
   new keyword-only parameters (`affordances`, `is_fragile`,
   `is_fixed`, `semantic_tags`, `text_query`), each with a falsy
   default. The positional six-arg ctor signature stays identical so
   every existing call site (six positional args) keeps working. The
   regression test `tests/regression/test_detected_object_compat.py`
   pins both the positional ctor AND the exact `__slots__` set so
   future drift is caught at CI time.
5. **Frame buffer.** Phase B adds `FrameBuffer` (a
   `collections.deque(maxlen=...)`-based ring) to the perception
   facade. `update_frames` writes atomically to BOTH the buffer AND
   the existing `_last_rgb` / `_last_depth` single-slot cache so a
   reader inspecting the buffer length cannot see a state where
   `_last_rgb is None` (closes peer-review H8). `stop()` clears both
   sides together. A property test in
   `tests/property/test_frame_buffer_invariants.py` asserts the
   invariant `len(buffer) > 0 implies _last_rgb is not None`.
6. **Atomic widening guarantee.** All five new `DetectedObject`
   fields default to falsy values (`()`, `False`, `None`) so YOLO
   output emits structurally identical objects: a downstream
   consumer that does not yet know about the new fields sees no
   change in behaviour.

## Consequences

* **Positive.** A clean detector-kind switch keeps the YOLO default
  untouched while enabling the Gemini open-vocabulary path behind an
  opt-in extra. `DetectedObject` gains the affordance / fragility
  metadata the safety chain (ADR-0009) needs without breaking any
  existing caller.
* **Negative.** Every future detector backend must implement the
  full protocol surface (no "optional" shortcuts). The
  `__slots__` widening is now load-bearing for pickle / wheel-size
  guarantees; the regression test must be kept in sync.
* **Risks.** Gemini may return semantic tags YOLO never produces;
  Phase B / state extractor explicitly ignores unknown tags unless
  `cfg.arm_task` opts in (plan §11 R-G6).

## Alternatives considered

* **Subclass `DetectedObject`** into `OpenVocabDetectedObject`.
  Rejected because state extractor + planner code branches on
  isinstance ladders, and a heterogeneous list of detection objects
  forces every consumer into narrowing.
* **Add an `extra: dict[str, Any]` bag** to `DetectedObject`.
  Rejected on the no-untyped-values rule (`CLAUDE.md`). Typed
  optional fields scale better.

## References

* Peer-review BLOCKER B4 and HIGH H2 (resolution table, plan §12).
* Plan section 3 Phase A + Phase B for the implementation timing.
