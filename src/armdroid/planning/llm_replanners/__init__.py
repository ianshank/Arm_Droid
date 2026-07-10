"""LLM-backed replanner implementations for the arm planning subsystem.

Public surface:
    * :func:`build_llm_replanner` -- construct the backend named by config.
    * registry helpers (:func:`register_llm_replanner`,
      :func:`available_llm_replanners`, ...) -- inspect or extend the set
      of backends, including out-of-tree plugins registered under the
      ``armdroid.llm_replanners`` entry-point group.
"""

from __future__ import annotations

from armdroid.planning.llm_replanners.factory import build_llm_replanner
from armdroid.planning.llm_replanners.registry import (
    available_llm_replanners,
    get_llm_replanner,
    load_llm_replanner_plugins,
    register_llm_replanner,
)

__all__ = [
    "available_llm_replanners",
    "build_llm_replanner",
    "get_llm_replanner",
    "load_llm_replanner_plugins",
    "register_llm_replanner",
]
