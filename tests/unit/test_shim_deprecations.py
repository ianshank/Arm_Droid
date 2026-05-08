"""Phase 3: verify DeprecationWarning is emitted from each legacy shim module.

Shims are already loaded by earlier imports in the test session, so we must
pop them from sys.modules and reimport them inside a warnings.catch_warnings()
context with ``simplefilter("always")`` to observe the per-import warning.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest


def _reimport_with_warning(module_name: str) -> pytest.WarningsChecker:
    """Remove *module_name* from sys.modules and reimport it, returning warnings."""
    sys.modules.pop(module_name, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module(module_name)
    return caught  # type: ignore[return-value]


class TestShimDeprecationWarnings:
    """Each legacy shim emits a DeprecationWarning on import."""

    def test_protocols_shim_warns(self) -> None:
        caught = _reimport_with_warning("armdroid.protocols")
        messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any("armdroid.protocols" in m for m in messages), messages

    def test_factory_shim_warns(self) -> None:
        caught = _reimport_with_warning("armdroid.factory")
        messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any("armdroid.factory" in m for m in messages), messages

    def test_orchestrator_shim_warns(self) -> None:
        caught = _reimport_with_warning("armdroid.orchestrator")
        messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any("armdroid.orchestrator" in m for m in messages), messages

    def test_esp32_json_driver_shim_warns(self) -> None:
        caught = _reimport_with_warning("armdroid.hardware.esp32_json_driver")
        messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any("esp32_json_driver" in m for m in messages), messages
