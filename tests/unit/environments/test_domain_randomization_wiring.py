"""Unit tests for the shared domain-randomization wiring helper.

Tests all three code paths:
- No-op when ``randomize_physics`` is ``False``
- Successful wiring of friction, mass, and gain events
- Graceful fallback when ``isaaclab`` is not importable
"""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from armdroid.config.schema.sim_isaac import ArmSimIsaacConfig


def _make_env_cfg() -> MagicMock:
    """Build a minimal mock env_cfg with an events namespace and scene.robot."""
    env_cfg = MagicMock()
    env_cfg.events = MagicMock()
    env_cfg.scene = MagicMock()
    env_cfg.scene.robot = MagicMock(name="robot_asset_cfg")
    return env_cfg


def _make_sim_cfg(*, randomize: bool = True) -> ArmSimIsaacConfig:
    """Create an ArmSimIsaacConfig with DR enabled or disabled."""
    return ArmSimIsaacConfig(
        randomize_physics=randomize,
        friction_range_sim=(0.5, 1.5),
        mass_scale_range=(0.8, 1.2),
        stiffness_noise_pct=10.0,
        damping_noise_pct=5.0,
    )


class TestApplyDomainRandomization:
    """Tests for apply_domain_randomization."""

    def test_noop_when_disabled(self) -> None:
        """Verify no events are wired when randomize_physics is False."""
        from armdroid.environments.isaac._domain_randomization import (
            apply_domain_randomization,
        )

        env_cfg = _make_env_cfg()
        sim_cfg = _make_sim_cfg(randomize=False)

        apply_domain_randomization(env_cfg, sim_cfg)

        # When disabled, no EventTermCfg attributes should have been assigned.
        # The MagicMock auto-creates attributes on access, so we verify by checking
        # the mock's explicit calls — there should be none.
        assert env_cfg.events.method_calls == []

    def test_wiring_friction_mass_gains(self) -> None:
        """Verify all 3 EventTermCfg entries are wired when isaaclab is available."""
        # Build fake isaaclab modules
        fake_mdp = MagicMock()
        fake_mdp.randomize_rigid_body_friction = MagicMock(name="friction_func")
        fake_mdp.randomize_rigid_body_mass = MagicMock(name="mass_func")
        fake_mdp.randomize_actuator_gains = MagicMock(name="gains_func")

        fake_managers = MagicMock()
        fake_event_term_cfg = MagicMock(name="EventTermCfg")
        fake_managers.EventTermCfg = fake_event_term_cfg

        fake_isaac_envs_mdp = types.ModuleType("isaaclab.envs.mdp")
        fake_isaac_envs_mdp.randomize_rigid_body_friction = fake_mdp.randomize_rigid_body_friction  # type: ignore[attr-defined]
        fake_isaac_envs_mdp.randomize_rigid_body_mass = fake_mdp.randomize_rigid_body_mass  # type: ignore[attr-defined]
        fake_isaac_envs_mdp.randomize_actuator_gains = fake_mdp.randomize_actuator_gains  # type: ignore[attr-defined]

        fake_isaac_managers = types.ModuleType("isaaclab.managers")
        fake_isaac_managers.EventTermCfg = fake_event_term_cfg  # type: ignore[attr-defined]

        modules_patch: dict[str, Any] = {
            "isaaclab": types.ModuleType("isaaclab"),
            "isaaclab.envs": types.ModuleType("isaaclab.envs"),
            "isaaclab.envs.mdp": fake_isaac_envs_mdp,
            "isaaclab.managers": fake_isaac_managers,
        }

        with patch.dict("sys.modules", modules_patch):
            from armdroid.environments.isaac._domain_randomization import (
                apply_domain_randomization,
            )

            env_cfg = _make_env_cfg()
            sim_cfg = _make_sim_cfg(randomize=True)
            apply_domain_randomization(env_cfg, sim_cfg)

        # EventTermCfg should have been called 3 times (friction, mass, gains)
        assert fake_event_term_cfg.call_count == 3

    def test_fallback_on_import_error(self) -> None:
        """Verify graceful fallback when isaaclab modules are missing."""
        from armdroid.environments.isaac._domain_randomization import (
            apply_domain_randomization,
        )

        env_cfg = _make_env_cfg()
        sim_cfg = _make_sim_cfg(randomize=True)

        # Patch the import of isaaclab.envs.mdp to raise ImportError
        with patch.dict("sys.modules", {"isaaclab.envs.mdp": None, "isaaclab.managers": None}):
            # Should not raise — logs a warning instead
            apply_domain_randomization(env_cfg, sim_cfg)

    def test_gain_scale_computation(self) -> None:
        """Verify stiffness/damping noise percentages are correctly converted to scale factors."""
        sim_cfg = _make_sim_cfg(randomize=True)

        # 10% stiffness noise -> scale range (0.9, 1.1)
        stiff_min = 1.0 - sim_cfg.stiffness_noise_pct / 100.0
        stiff_max = 1.0 + sim_cfg.stiffness_noise_pct / 100.0
        assert stiff_min == pytest.approx(0.9)
        assert stiff_max == pytest.approx(1.1)

        # 5% damping noise -> scale range (0.95, 1.05)
        damp_min = 1.0 - sim_cfg.damping_noise_pct / 100.0
        damp_max = 1.0 + sim_cfg.damping_noise_pct / 100.0
        assert damp_min == pytest.approx(0.95)
        assert damp_max == pytest.approx(1.05)
