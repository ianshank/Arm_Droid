"""Serial port discovery and probing for the ESP32 JSON driver."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import TYPE_CHECKING, Any

from armdroid.logging.setup import get_logger
from armdroid.protocols import ArmDriverError

if TYPE_CHECKING:
    from armdroid.config.schema import ArmConfig

_log = get_logger(__name__)


def open_port_blocking(device: str, cfg: ArmConfig, serial_module: Any) -> Any:
    """Open *device* as a ``serial.Serial`` instance (blocking, run in a thread pool).

    Args:
        device: OS device path (e.g. ``/dev/ttyUSB0``, ``COM3``).
        cfg: Arm configuration for baud rate and timeouts.
        serial_module: The imported ``serial`` module.

    Returns:
        An open :class:`serial.Serial` instance.
    """
    return serial_module.Serial(
        port=device,
        baudrate=cfg.transport.serial_baud,
        timeout=cfg.transport.connect_timeout_s,
        write_timeout=cfg.transport.command_timeout_s,
    )


def probe_port_blocking(device: str, cfg: ArmConfig, serial_module: Any) -> str | None:
    """Open *device*, send a ping, wait for a matching ack.

    Returns *device* on success, ``None`` if the port doesn't respond like
    the firmware. Always closes the port — the caller re-opens it in
    ``connect()``.

    Args:
        device: Serial port device path to probe.
        cfg: Arm configuration for baud rate and timeouts.
        serial_module: The imported ``serial`` module.

    Returns:
        *device* if the firmware responded, else ``None``.
    """
    port: Any | None = None
    try:
        port = serial_module.Serial(
            port=device,
            baudrate=cfg.transport.serial_baud,
            timeout=cfg.transport.connect_timeout_s,
            write_timeout=cfg.transport.command_timeout_s,
        )
        ping_msg = (
            json.dumps(
                {"t": "cmd", "id": 1, "ts": time.monotonic(), "cmd": "ping"},
                separators=(",", ":"),
            )
            + "\n"
        )
        port.write(ping_msg.encode("ascii"))
        port.flush()
        deadline = time.monotonic() + cfg.transport.connect_timeout_s
        while time.monotonic() < deadline:
            line = port.readline()
            if not line:
                continue
            try:
                msg = json.loads(line.decode("ascii", errors="replace").strip())
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if msg.get("t") == "ack" and msg.get("id") == 1:
                return device
            if msg.get("t") == "evt" and msg.get("kind") == "boot":
                # Firmware just booted; re-issue the ping after a beat.
                port.write(ping_msg.encode("ascii"))
                port.flush()
        return None
    finally:
        if port is not None:
            with contextlib.suppress(Exception):  # pragma: no cover
                port.close()


async def autodetect_port(
    cfg: ArmConfig,
    serial_module: Any,
    list_ports_module: Any,
) -> str:
    """Probe USB serial ports for the firmware boot signature in parallel.

    Sends a ``ping`` to each candidate and binds the first one that returns
    a valid ``ack`` within ``connect_timeout_s``.

    Args:
        cfg: Arm configuration (transport sub-model for hints, exclusions,
            and probe concurrency).
        serial_module: The imported ``serial`` module.
        list_ports_module: The imported ``serial.tools.list_ports`` module.

    Returns:
        The first device path that responded with a valid firmware ack.

    Raises:
        ArmDriverError: If no candidates are found or none responded.
    """
    excluded = set(cfg.transport.exclude_ports)
    hints = [h.lower() for h in cfg.transport.usb_vid_pid_hints]
    candidates: list[str] = []
    for port_info in list_ports_module.comports():
        device = str(port_info.device)
        if device in excluded:
            continue
        if hints:
            hwid = (port_info.hwid or "").lower()
            if not any(h in hwid for h in hints):
                continue
        candidates.append(device)

    if not candidates:
        msg = (
            "Port autodetect found no candidate USB serial ports. "
            "Plug the ESP32 in or set cfg.arm.transport.serial_port "
            "to an explicit path."
        )
        raise ArmDriverError(msg)

    _log.info("esp32_json_autodetect_candidates", candidates=candidates)
    sem = asyncio.Semaphore(cfg.transport.autodetect_probe_concurrency)

    async def _probe(device: str) -> str | None:
        async with sem:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(probe_port_blocking, device, cfg, serial_module),
                    timeout=cfg.transport.connect_timeout_s,
                )
            except (TimeoutError, ArmDriverError, OSError) as exc:
                _log.debug(
                    "esp32_json_autodetect_probe_failed",
                    port=device,
                    error=str(exc),
                )
                return None

    results = await asyncio.gather(*(_probe(d) for d in candidates))
    for result in results:
        if result is not None:
            _log.info("esp32_json_autodetect_bound", port=result)
            return result

    msg = (
        f"Port autodetect probed {len(candidates)} candidate(s) but none "
        "responded with a valid firmware ack. Set serial_port explicitly "
        "or check that the firmware is flashed."
    )
    raise ArmDriverError(msg)


async def resolve_port(
    cfg: ArmConfig,
    serial_module: Any,
    list_ports_module: Any,
) -> str:
    """Return the configured serial port, resolving ``"auto"`` via probing.

    Args:
        cfg: Arm configuration (``transport.serial_port`` may be ``"auto"``).
        serial_module: The imported ``serial`` module.
        list_ports_module: The imported ``serial.tools.list_ports`` module.
            May be ``None``; raises :exc:`ArmDriverError` if ``"auto"`` is
            requested and this is ``None``.

    Returns:
        Resolved device path string.

    Raises:
        ArmDriverError: If ``"auto"`` is requested but port probing fails
            or ``list_ports_module`` is not available.
    """
    configured = cfg.transport.serial_port
    if configured != "auto":
        return configured
    if list_ports_module is None:  # pragma: no cover - hardware extra
        msg = (
            "serial.tools.list_ports not available — install pyserial >= 3.5"
            " to enable port autodetect."
        )
        raise ArmDriverError(msg)
    return await autodetect_port(cfg, serial_module, list_ports_module)


__all__ = ["autodetect_port", "open_port_blocking", "probe_port_blocking", "resolve_port"]
