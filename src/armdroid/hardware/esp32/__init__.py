"""ESP32 JSON driver subpackage.

Public surface: :class:`Esp32JsonDriver`.

Sub-modules:

* :mod:`.driver`    — orchestrator class and module-level constants
* :mod:`.framing`   — wire-frame decode and pending-reply bookkeeping
* :mod:`.portfinder` — serial port discovery and probing
* :mod:`.validator` — local command validation and velocity-anchor logic
"""

from armdroid.hardware.esp32.driver import Esp32JsonDriver as Esp32JsonDriver

__all__ = ["Esp32JsonDriver"]
