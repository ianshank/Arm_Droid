"""HIL conftest — discover real ESP32 port + skip module if absent.

These tests run only when an ESP32 with armdroid firmware flashed is
connected to the host. The discovery here mirrors the host-side
``Esp32JsonDriver._autodetect_port`` logic but exits early so the
whole module skips cleanly when no hardware is attached.

Run the HIL suite explicitly with:

    pytest tests/hardware -m hardware

CI keeps these gated until a self-hosted runner with the hardware is
provisioned; the tests are written to PROTOCOL.md so they are
ready to run as soon as a runner is available.
"""

from __future__ import annotations

import os

import pytest

from armdroid.config.schema import ArmSettings


def _try_discover_esp32() -> str | None:
    """Return the device path of an attached ESP32 + armdroid firmware.

    Returns None — and the test gets skipped — when the host is not set
    up for HIL. We require an explicit opt-in via ``ARMDROID_HIL_RUN=1``
    so default ``pytest tests/`` runs do NOT attempt to talk to whatever
    serial device happens to be plugged in (which is almost never the
    armdroid firmware on a dev box).
    """
    if os.environ.get("ARMDROID_HIL_RUN") != "1":
        return None
    try:
        from serial.tools import list_ports
    except ImportError:
        return None

    explicit = os.environ.get("ARMDROID_ARM__TRANSPORT__SERIAL_PORT")
    if explicit and explicit != "auto":
        return explicit

    candidates = [info.device for info in list_ports.comports()]
    return candidates[0] if candidates else None


@pytest.fixture(scope="session")
def hil_port() -> str:
    """Yield a real serial port path or skip the test."""
    port = _try_discover_esp32()
    if port is None:
        pytest.skip("No ESP32 + armdroid firmware detected (HIL test skipped).")
    return port


@pytest.fixture
def hil_settings(hil_port: str) -> ArmSettings:
    """ArmSettings configured for real hardware on the discovered port."""
    base = ArmSettings(mock_hardware=False)
    # Build a fresh config tree via Pydantic's model_copy so we never
    # mutate shared model instances (which can bleed between tests).
    patched_transport = base.arm.transport.model_copy(
        update={"serial_port": hil_port}
    )
    patched_arm = base.arm.model_copy(update={"transport": patched_transport})
    return base.model_copy(update={"arm": patched_arm})


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-mark every test in this directory with `hardware`."""
    for item in items:
        if "tests/hardware" in str(item.fspath).replace("\\", "/"):
            item.add_marker(pytest.mark.hardware)
