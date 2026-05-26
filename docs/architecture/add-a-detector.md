# How to add a perception detector

Companion guide to ADR-0008. Use this checklist when you want to plug
a new object-detector backend (Gemini ER 1.6, Owl-ViT, Grounding-DINO,
...) into the perception facade alongside the YOLO default.

## Checklist

1. **Implement `ObjectDetectorProtocol`** in a new module under
   `src/armdroid/perception/<backend_name>/detector.py`.
   * Methods: `load_model()`, `detect(rgb) -> list[DetectedObject]`,
     `detect_text(rgb, prompt) -> list[DetectedObject]`,
     `infer_affordances(detections, rgb) -> list[DetectedObject]`.
   * Every method is **concretely required**; there is no "optional
     method" semantic in `@runtime_checkable Protocol`. Closed-vocab
     detectors typically return `[]` for `detect_text` and the
     unmodified list for `infer_affordances`.

2. **Lazy SDK import.** Mirror `AnthropicReplanner._import_sdk` for
   any heavy dependency. Raise a backend-specific
   `*SDKMissingError` with an install hint when the optional extra
   is missing.

3. **Register in the detector registry.** Phase B introduces
   `src/armdroid/perception/detector_registry.py` with entry-point
   group `armdroid.perception_detectors`. Add a registration in code
   and a matching `[project.entry-points."armdroid.perception_detectors"]`
   entry in `pyproject.toml` so the entry-point mirror regression
   stays green.

4. **Configure via `ArmPerceptionConfig`.** All knobs (model id,
   thresholds, API key env var, request timeout, max retries) route
   through Pydantic fields with `ge`/`le`/`gt`/`Literal` bounds. No
   hardcoded values.

5. **Emit `DetectedObject` with sensible defaults.** Closed-vocab
   detectors construct `DetectedObject` positionally (six args);
   open-vocab detectors additionally set the keyword-only
   `affordances`, `is_fragile`, `is_fixed`, `semantic_tags`,
   `text_query` fields. The downstream state extractor ignores
   unknown semantic tags by default.

6. **Telemetry.** Use the Phase A SPAN constants
   `SPAN_PERCEPTION_DETECT`, `SPAN_PERCEPTION_DETECT_TEXT`,
   `SPAN_PERCEPTION_SCENE_REASON`,
   `SPAN_PERCEPTION_FRAME_BUFFER_APPEND`. Structured-log event
   names follow `<backend>_<action>_<status>` (see
   `gemini_detector_initialised`, `gemini_detect_text_failed`).

7. **Tests.**
   * Unit: `tests/unit/perception/test_<backend>_detector.py` using
     `tests/helpers/fake_llm_sdk` (or a backend-shaped helper).
   * Contract: parametrise
     `tests/contract/test_object_detector_contract.py` to include
     the new backend.
   * Regression:
     `tests/regression/test_yolo_default_path.py` keeps the YOLO
     default path byte-identical.

## DetectedObject widening contract

ADR-0008 pins the `__slots__` tuple so future drift would silently
break wheel size / pickle compatibility.
`tests/regression/test_detected_object_compat.py` asserts the set
membership. If you need to add a new metadata field, extend
`__slots__` and `__init__` together AND extend the regression test's
expected tuple in the same PR.
