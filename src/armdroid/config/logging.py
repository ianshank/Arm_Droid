"""Logging configuration model for armdroid."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """Logging configuration.

    Attributes:
        level: Minimum log level emitted (DEBUG/INFO/WARNING/ERROR/CRITICAL).
        format: Output format — ``console`` for human-readable dev, ``json`` for prod.
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Minimum log level emitted.",
    )
    format: Literal["console", "json"] = Field(
        default="console",
        description="Renderer format.",
    )
