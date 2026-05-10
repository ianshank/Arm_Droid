"""Isaac Lab task vendor tree.

Vendored from MuammerBay/isaac_so_arm101 (BSD-3-Clause). See
``THIRD_PARTY_NOTICES.md`` and the per-file copyright headers.

Module top-level does NOT eagerly import the vendored task code —
``armdroid.environments.isaac.reach.SoArmReachIsaacEnv`` triggers the
import lazily in ``_ensure_built()`` so default installs without
isaaclab can still load this package.
"""

from __future__ import annotations

__all__: list[str] = []
