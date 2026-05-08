"""Unit-test-level fixtures for armdroid.

Resets structlog state between tests to prevent processor-chain warnings
from leaking across test boundaries (e.g. the ``format_exc_info``
UserWarning that fires when structlog is re-configured mid-session).
"""

from __future__ import annotations

import pytest
import structlog


@pytest.fixture(autouse=True)
def _reset_structlog() -> None:
    """Reset structlog to its default (unconfigured) state before each unit test.

    Without this, calling ``configure_logging()`` in one test permanently
    mutates the global structlog processor chain. Subsequent tests that
    also call ``configure_logging()`` trigger:
        UserWarning: Remove `format_exc_info` from your processor chain
    because the chain now has duplicate processors.
    """
    structlog.reset_defaults()
