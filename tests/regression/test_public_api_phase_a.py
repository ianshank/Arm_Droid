"""Regression: Phase A public-API surface additions.

Pins that the new value objects and protocols (``ArmAction``,
``SceneInsight``, ``Verdict``, ``InteractionEvent``,
``VisionLanguageAgentProtocol``, ``HighLevelPlannerProtocol``,
``SafetyGuardProtocol``, ``InteractionSessionProtocol``) are reachable
through every documented public path:

  - ``armdroid`` (root convenience)
  - ``armdroid.api`` (SemVer-stable surface)
  - ``armdroid.domain`` (internal but documented re-export)

A future PR that adds these to ``armdroid.domain.state`` /
``armdroid.domain.protocols`` without lifting them through the
re-export chain will trip this test, surfacing the discoverability gap
at CI time instead of when a consumer hits ``ImportError``.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression


_PHASE_A_PUBLIC_NAMES: tuple[str, ...] = (
    # Value objects
    "ArmAction",
    "SceneInsight",
    "Verdict",
    "InteractionEvent",
    # Protocols
    "VisionLanguageAgentProtocol",
    "HighLevelPlannerProtocol",
    "SafetyGuardProtocol",
    "InteractionSessionProtocol",
)


def test_root_package_exports_all_phase_a_names() -> None:
    import armdroid

    for name in _PHASE_A_PUBLIC_NAMES:
        assert hasattr(armdroid, name), f"armdroid.{name} missing"
        assert name in armdroid.__all__, f"{name} missing from armdroid.__all__"


def test_api_exports_all_phase_a_names() -> None:
    import armdroid.api as api

    for name in _PHASE_A_PUBLIC_NAMES:
        assert hasattr(api, name), f"armdroid.api.{name} missing"
        assert name in api.__all__, f"{name} missing from armdroid.api.__all__"


def test_domain_exports_all_phase_a_names() -> None:
    import armdroid.domain as domain

    for name in _PHASE_A_PUBLIC_NAMES:
        assert hasattr(domain, name), f"armdroid.domain.{name} missing"
        assert name in domain.__all__, f"{name} missing from armdroid.domain.__all__"


def test_reexports_are_identity_links_not_copies() -> None:
    """The three import paths must point at the SAME object so
    ``isinstance`` checks against any of them work uniformly."""
    import armdroid
    import armdroid.api as api
    import armdroid.domain as domain
    import armdroid.domain.protocols as protocols_mod
    import armdroid.domain.state as state_mod

    for name in ("ArmAction", "Verdict", "SceneInsight", "InteractionEvent"):
        canonical = getattr(state_mod, name)
        assert getattr(armdroid, name) is canonical
        assert getattr(api, name) is canonical
        assert getattr(domain, name) is canonical

    for name in (
        "VisionLanguageAgentProtocol",
        "HighLevelPlannerProtocol",
        "SafetyGuardProtocol",
        "InteractionSessionProtocol",
    ):
        canonical = getattr(protocols_mod, name)
        assert getattr(armdroid, name) is canonical
        assert getattr(api, name) is canonical
        assert getattr(domain, name) is canonical
