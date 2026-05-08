"""PlatformIO pre-build hook: regenerate config_generated.h before each build.

Invoked by the ``extra_scripts = pre:gen_config_pre.py`` line in
platformio.ini. We shell out to the canonical generator script so there
is exactly one code path that knows how to render the header.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# The PlatformIO pre-build environment exposes `Import` for fetching the
# project env, but we deliberately avoid using it — this script is a
# zero-config side-car that just runs the canonical generator. Keeping it
# simple lets us also call it from other build systems later.

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "scripts" / "gen_firmware_config.py"


def main() -> int:
    if not GENERATOR.exists():
        sys.stderr.write(
            f"[gen_config_pre] generator not found at {GENERATOR}\n"
        )
        return 1
    rc = subprocess.run(
        [sys.executable, str(GENERATOR)],
        check=False,
        cwd=str(REPO_ROOT),
    ).returncode
    if rc != 0:
        sys.stderr.write(
            f"[gen_config_pre] generator failed with exit code {rc}\n"
        )
    return rc


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
