"""Tests for armdroid.logging.setup."""

from __future__ import annotations

import structlog

from armdroid.config.logging import LoggingConfig
from armdroid.logging.setup import configure_logging, get_logger


def test_get_logger_returns_bound_logger() -> None:
    """get_logger returns a structlog BoundLogger that can emit events."""
    log = get_logger("armdroid.test")
    assert log is not None
    log.info("smoke_event", marker="ok")


def test_configure_logging_console_format() -> None:
    """configure_logging accepts a console-format LoggingConfig."""
    cfg = LoggingConfig(level="DEBUG", format="console")
    configure_logging(cfg)
    log = get_logger("armdroid.test")
    log.debug("after_console_configure")


def test_configure_logging_json_format() -> None:
    """configure_logging accepts a json-format LoggingConfig."""
    cfg = LoggingConfig(level="WARNING", format="json")
    configure_logging(cfg)
    log = get_logger("armdroid.test")
    log.warning("after_json_configure")


def test_configure_logging_with_robot_id() -> None:
    """robot_id is bound into structlog contextvars."""
    cfg = LoggingConfig()
    configure_logging(cfg, robot_id="armdroid-test-1")
    bound = structlog.contextvars.get_contextvars()
    assert bound.get("robot_id") == "armdroid-test-1"
