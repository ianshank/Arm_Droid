"""PlatformIO pre-build hook: regenerate config_generated.h before each build.

Invoked by the ``extra_scripts = pre:gen_config_pre.py`` line in
platformio.ini. We shell out to the canonical generator script so there
is exactly one code path that knows how to render the header.

Note on SCons execution context
--------------------------------
PlatformIO executes extra_scripts files via SCons ``exec()``, which means
``__file__`` is **not** defined. We resolve the repo root from the SCons
``env["PROJECT_DIR"]`` variable instead, which is always set by PlatformIO
to the directory that contains ``platformio.ini``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ``Import`` and ``env`` are injected as builtins by the SCons/PlatformIO
# execution environment.  We import env through the SCons builtins API so
# that the module is also importable (and testable) outside of PlatformIO.
try:
    Import("env")  # type: ignore[name-defined]  # noqa: F821
    _PROJECT_DIR = Path(env["PROJECT_DIR"]).resolve()  # type: ignore[name-defined]  # noqa: F821
except NameError:
    # Running outside PlatformIO (e.g. direct ``python gen_config_pre.py``).
    # Fall back to the directory that contains this script.
    _PROJECT_DIR = Path(__file__).resolve().parent

# firmware/arm_esp32/ -> firmware/ -> repo root
REPO_ROOT = _PROJECT_DIR.parents[1]
GENERATOR = REPO_ROOT / "scripts" / "gen_firmware_config.py"


def main() -> int:
    """Run the generator; return the process exit code."""
    if not GENERATOR.exists():
        sys.stderr.write(
            f"[gen_config_pre] generator not found at {GENERATOR}\n"
        )
        return 1
    rc = subprocess.run(  # noqa: S603
        [sys.executable, str(GENERATOR)],
        check=False,
        cwd=str(REPO_ROOT),
    ).returncode
    if rc != 0:
        sys.stderr.write(
            f"[gen_config_pre] generator failed with exit code {rc}\n"
        )
    return rc


# Entry point when invoked from PlatformIO's SCons context (extra_scripts).
# PlatformIO executes the file body directly, so ``main()`` runs here.
# When run directly as a script, the ``if __name__ == "__main__"`` block
# below handles the call so main() does not execute twice.
if __name__ != "__main__":
    main()

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
