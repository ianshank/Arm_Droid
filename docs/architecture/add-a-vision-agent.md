# How to add a Vision-Language Agent

Companion guide to ADR-0007. Use this checklist when you want to plug
a new VLA backend (Gemini ER 1.6 successor, GR00T, OpenVLA, ...) into
armdroid alongside the SAC / RSL-RL PPO controllers.

## Checklist

1. **Implement `VisionLanguageAgentProtocol`** in a new module under
   `src/armdroid/control/vla/<backend_name>.py`.
   * Methods: `build_vla(env)`, `act(rgb, depth, instruction, state)
     -> ArmAction`, `is_vla_built`, `is_vla_loaded`. Keep method
     names disjoint from `ArmRLAgentProtocol` so `isinstance`
     dispatch in `ArmController` stays unambiguous.
   * Mirror `AnthropicReplanner` for lazy SDK import, structured
     logging, and retry semantics.

2. **Lazy SDK import.** Wrap the SDK import in `_import_sdk(self)
   -> Any` that raises a backend-specific `*SDKMissingError` with an
   install hint when the optional extra is missing. Pattern:
   ```python
   def _import_sdk(self) -> Any:
       try:
           import google.genai as sdk
       except ImportError as exc:
           msg = "Install `google-genai` via `pip install -e \".[gemini-vla]\"`."
           raise VlaSDKMissingError(msg) from exc
       return sdk
   ```

3. **Register in the VLA registry.** Add an entry to
   `src/armdroid/control/vla/registry.py` and a matching
   `[project.entry-points."armdroid.vision_agents"]` block in
   `pyproject.toml` so the entry-point mirror regression stays
   green.

4. **Configure via `ArmVlaConfig`.** Every numeric / string knob
   (model id, request timeout, action clip, frame stack, API key env
   var) routes through `ArmVlaConfig`. No hardcoded values inside
   the backend module.

5. **Telemetry.** Wrap each call site with the existing
   `SPAN_VLA_BUILD`, `SPAN_VLA_ACT`, `SPAN_VLA_PARSE` constants. Use
   `get_logger(__name__).info("event_name", phase=..., backend=...)`
   for structured logs.

6. **Tests.**
   * Unit: `tests/unit/control/test_<backend>_vla.py` using
     `tests/helpers/fake_llm_sdk.fake_gemini_sdk` (or a similarly
     shaped helper for non-Gemini providers).
   * Contract: parametrise `tests/contract/test_vision_agent_contract.py`
     to include the new backend.
   * Regression: assert
     `tests/regression/test_default_controller_path.py` still builds
     the SAC path when `cfg.arm_vla.enabled` is False (default).

## Disjoint-method invariant

`@runtime_checkable Protocol.isinstance` matches by method-name set
only. The
`tests/unit/domain/test_new_protocols.py::test_rl_agent_does_not_match_vla_protocol`
test pins the invariant. If you ever need to add a method to
`VisionLanguageAgentProtocol` that overlaps with
`ArmRLAgentProtocol`, the test will fail and the dispatch in
`ArmController` is no longer correct - prefer renaming over reuse.
