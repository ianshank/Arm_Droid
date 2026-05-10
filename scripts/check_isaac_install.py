"""Pre-install / pre-boot probe for the Isaac Sim 5.1 path.

Reports four things the ``[isaac]`` extra needs at runtime:

1. Python version (Isaac Lab 2.3 requires 3.11; isaacsim wheels do not
   ship 3.12+ tags as of 2026-05).
2. NVIDIA driver version (Isaac Sim 5.1 minimum: 535).
3. Per-GPU compute capability. Isaac Sim 5.1 / Isaac Lab 2.3 officially
   require Ampere (compute capability ``8.0+``) or newer for the RTX
   renderer + PhysX-GPU paths. Older architectures (Pascal 6.x, Volta
   7.0, Turing 7.5) may install but crash at AppLauncher boot or fall
   back to CPU PhysX (~50x slower for the parallel-env training path).
4. Which Isaac / RL packages are currently importable.

This script is **safe to run on any machine** - it never imports
``isaaclab`` or ``isaacsim``, only ``torch`` (already in armdroid's
base deps) and the standard library, plus an ``nvidia-smi`` shell-out.

Exit code:
    0  - Compatible-or-better; safe to ``pip install -e ".[isaac]"``.
    1  - Hard incompatibility (no GPU, driver too old, no torch).
    2  - Soft incompatibility (Pascal/Volta/Turing - may work in CPU
         PhysX fallback, RTX renderer disabled). Caller should weigh
         the cost of the ~9-15 GB install before proceeding.

Usage::

    .venv/Scripts/python.exe scripts/check_isaac_install.py

    # Use exit code in CI / install scripts:
    python scripts/check_isaac_install.py || echo "Skipping [isaac]"
"""

from __future__ import annotations

import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass

# Minimum NVIDIA driver version per Isaac Sim 5.1 official requirements.
# See: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html
# (``latest`` slug avoids version-rot when NVIDIA rotates the per-release
# doc URLs.)
ISAAC_SIM_MIN_DRIVER_MAJOR = 535

# Compute capability thresholds. Ampere (compute 8.0) is the official
# Isaac Sim 5.1 minimum; older architectures may load but crash or fall
# back to CPU.
COMPUTE_CAPABILITY_AMPERE = (8, 0)

# Python version pin for Isaac Lab 2.3.
ISAAC_LAB_PYTHON = (3, 11)

# Packages required by armdroid's [isaac] extra.
ISAAC_PACKAGES = (
    "isaaclab",
    "isaacsim",
    "isaacsim_kernel",
    "isaaclab_rl",
    "rsl_rl",
)
# Packages that should always be present from base [dev].
BASE_PACKAGES = ("torch", "numpy", "gymnasium")


@dataclass(frozen=True)
class GpuInfo:
    """Single-GPU info as queried from nvidia-smi + torch."""

    index: int
    name: str
    driver_version: str
    memory_mib: int
    compute_capability: tuple[int, int] | None


def _check_python() -> tuple[bool, str]:
    """Return (ok, message) for the Python version."""
    cur = sys.version_info[:2]
    expected = ISAAC_LAB_PYTHON
    if cur == expected:
        return True, f"Python {cur[0]}.{cur[1]} (matches Isaac Lab pin)"
    if cur < expected:
        return False, f"Python {cur[0]}.{cur[1]} too old (need {expected[0]}.{expected[1]})"
    return False, (
        f"Python {cur[0]}.{cur[1]} newer than Isaac Lab pin "
        f"({expected[0]}.{expected[1]}); isaacsim wheels may not have "
        f"matching tags"
    )


def _query_nvidia_smi() -> Sequence[GpuInfo] | None:
    """Return a tuple of GpuInfo or None if nvidia-smi is missing/failed.

    Caveat: in mixed Tesla + GeForce driver setups on Windows,
    nvidia-smi only sees GPUs sharing the driver branch its host
    Tesla/GeForce binary was built for. Use _query_windows_adapters() as
    a second source to surface the full hardware picture.
    """
    if not shutil.which("nvidia-smi"):
        return None
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=10)
    except (subprocess.SubprocessError, OSError):
        return None
    rows: list[GpuInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            rows.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    driver_version=parts[2],
                    memory_mib=int(parts[3]),
                    compute_capability=None,  # filled in by torch lookup
                )
            )
        except ValueError:
            continue
    return tuple(rows) if rows else None


# Name-prefix -> compute capability heuristic. Used when neither
# torch.cuda.get_device_capability nor nvidia-smi --query-gpu=compute_cap
# can answer (multi-driver Windows boxes, CPU-only torch). Pattern
# match is case-insensitive prefix + first hit wins; ranges below
# Ampere are intentionally listed so the verdict path can flag them.
_NAME_TO_COMPUTE_CAP: tuple[tuple[re.Pattern[str], tuple[int, int], str], ...] = (
    (re.compile(r"\brtx\s*50\d{2}", re.I), (12, 0), "Blackwell consumer"),
    (re.compile(r"\brtx\s*pro\s*\d", re.I), (12, 0), "Blackwell workstation"),
    (re.compile(r"\b(b100|b200|gb200)\b", re.I), (10, 0), "Blackwell datacenter"),
    (re.compile(r"\brtx\s*40\d{2}", re.I), (8, 9), "Ada Lovelace"),
    (re.compile(r"\b(l40|l4|l20)\b", re.I), (8, 9), "Ada Lovelace datacenter"),
    (re.compile(r"\brtx\s*30\d{2}", re.I), (8, 6), "Ampere consumer"),
    (re.compile(r"\b(a100|a30|a40|a10)\b", re.I), (8, 0), "Ampere datacenter"),
    (re.compile(r"\brtx\s*20\d{2}", re.I), (7, 5), "Turing"),
    (re.compile(r"\bquadro\s+rtx", re.I), (7, 5), "Turing workstation"),
    (re.compile(r"\b(t4|titan\s+rtx)\b", re.I), (7, 5), "Turing"),
    (re.compile(r"\b(v100|titan\s+v)\b", re.I), (7, 0), "Volta"),
    (re.compile(r"\b(p40|p100|p4|titan\s+xp)\b", re.I), (6, 1), "Pascal"),
    (re.compile(r"\bgtx\s*10\d{2}", re.I), (6, 1), "Pascal consumer"),
)


def _guess_compute_capability(name: str) -> tuple[tuple[int, int], str] | None:
    """Best-effort compute capability lookup from GPU name."""
    for pattern, cap, family in _NAME_TO_COMPUTE_CAP:
        if pattern.search(name):
            return cap, family
    return None


# Sentinel "no compute capability discoverable" value. Comparing
# (0, 0) to COMPUTE_CAPABILITY_AMPERE evaluates to False, so unknown
# GPUs are treated as not-Ampere-or-better in the verdict.
_UNKNOWN_CAPABILITY: tuple[int, int] = (0, 0)


def _effective_capability(gpu: GpuInfo) -> tuple[int, int]:
    """Return torch-confirmed cap, falling back to name-heuristic, then sentinel."""
    if gpu.compute_capability is not None:
        return gpu.compute_capability
    guess = _guess_compute_capability(gpu.name)
    if guess is not None:
        return guess[0]
    return _UNKNOWN_CAPABILITY


def _query_windows_adapters() -> Sequence[GpuInfo] | None:
    """Enumerate all display adapters via Windows CIM. Windows-only.

    Returns NVIDIA-only adapters. Driver version is the WDDM string
    (e.g. ``32.0.15.9621`` -> driver branch 582.78). AdapterRAM is
    capped at 4 GiB for legacy uint32 reasons; the value reported is
    informational only and may understate the real VRAM.
    """
    if platform.system() != "Windows":
        return None
    if not shutil.which("powershell"):
        return None
    ps_cmd = (
        "Get-CimInstance Win32_VideoController | "
        "Where-Object { $_.Name -match 'NVIDIA' } | "
        "Select-Object Name, DriverVersion, AdapterRAM | "
        "ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if not out.strip():
        return None
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return None
    rows: list[GpuInfo] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("Name", "")).strip()
        if not name:
            continue
        driver = str(entry.get("DriverVersion", "")).strip()
        ram_bytes = entry.get("AdapterRAM") or 0
        try:
            ram_mib = int(ram_bytes) // (1024 * 1024) if ram_bytes else 0
        except (TypeError, ValueError):
            ram_mib = 0
        rows.append(
            GpuInfo(
                index=index,
                name=name,
                driver_version=driver,
                memory_mib=ram_mib,
                compute_capability=None,
            )
        )
    return tuple(rows) if rows else None


def _merge_gpu_sources(
    nvidia_smi: Sequence[GpuInfo] | None,
    windows: Sequence[GpuInfo] | None,
) -> tuple[Sequence[GpuInfo], list[str]]:
    """Combine the two enumeration sources, de-duplicating by name.

    Returns the merged list plus a list of human-readable warnings
    about discrepancies (e.g. a card present in Windows that
    nvidia-smi missed). The merge prefers nvidia-smi entries (richer
    metadata) and falls back to Windows-only entries for cards
    nvidia-smi did not see.
    """
    warnings: list[str] = []
    if nvidia_smi is None and windows is None:
        return (), warnings
    if windows is None:
        return tuple(nvidia_smi or ()), warnings
    if nvidia_smi is None:
        warnings.append(
            "nvidia-smi missing or returned no GPUs; using Windows "
            "adapter enumeration only (no real-time VRAM / utilisation)."
        )
        return tuple(windows), warnings

    def _norm(name: str) -> str:
        # nvidia-smi reports e.g. "Tesla P40" while Windows reports
        # "NVIDIA Tesla P40"; collapse the prefix + whitespace so
        # cross-source matches dedupe.
        cleaned = re.sub(r"^\s*nvidia\s+", "", name, flags=re.I)
        return re.sub(r"\s+", " ", cleaned).strip().casefold()

    smi_names = {_norm(gpu.name) for gpu in nvidia_smi}
    extras = [gpu for gpu in windows if _norm(gpu.name) not in smi_names]
    if extras:
        names = ", ".join(gpu.name for gpu in extras)
        warnings.append(
            f"Windows reports {len(extras)} extra NVIDIA GPU(s) that "
            f"nvidia-smi did not enumerate: {names}. This typically "
            f"means the system mixes Tesla/data-center and GeForce "
            f"driver branches; nvidia-smi sees only one branch's cards. "
            f"Compute capability for the missing card(s) is inferred "
            f"from the device name."
        )
    # Re-index extras after the nvidia-smi list for stable display.
    base_count = len(nvidia_smi)
    rebased = [
        GpuInfo(
            index=base_count + i,
            name=gpu.name,
            driver_version=gpu.driver_version,
            memory_mib=gpu.memory_mib,
            compute_capability=gpu.compute_capability,
        )
        for i, gpu in enumerate(extras)
    ]
    return tuple(list(nvidia_smi) + rebased), warnings


@dataclass(frozen=True)
class TorchInfo:
    """torch build metadata relevant to the Isaac install path."""

    importable: bool
    version: str | None
    cuda_version: str | None
    cuda_runtime_available: bool


def _query_torch() -> TorchInfo:
    """Return a TorchInfo without raising even when torch is missing."""
    if importlib.util.find_spec("torch") is None:
        return TorchInfo(False, None, None, False)
    try:
        import torch
    except ImportError:
        return TorchInfo(False, None, None, False)
    return TorchInfo(
        importable=True,
        version=torch.__version__,
        cuda_version=torch.version.cuda,
        cuda_runtime_available=torch.cuda.is_available(),
    )


def _enrich_with_torch(gpus: Sequence[GpuInfo], torch_info: TorchInfo) -> Sequence[GpuInfo]:
    """Attach compute capability via torch when CUDA runtime is live."""
    if not torch_info.cuda_runtime_available:
        return gpus
    try:
        import torch
    except ImportError:
        return gpus
    enriched: list[GpuInfo] = []
    for gpu in gpus:
        try:
            cap = torch.cuda.get_device_capability(gpu.index)
            enriched.append(
                GpuInfo(
                    index=gpu.index,
                    name=gpu.name,
                    driver_version=gpu.driver_version,
                    memory_mib=gpu.memory_mib,
                    compute_capability=cap,
                )
            )
        except (RuntimeError, AssertionError):
            enriched.append(gpu)
    return tuple(enriched)


def _verdict_for_gpu(gpu: GpuInfo) -> tuple[int, str]:
    """Per-GPU verdict. Returns (severity, message). 0=ok, 1=hard, 2=soft."""
    cap = gpu.compute_capability
    inferred_family: str | None = None
    if cap is None:
        guess = _guess_compute_capability(gpu.name)
        if guess is not None:
            cap, inferred_family = guess
    if cap is None:
        return 2, (
            f"GPU {gpu.index} ({gpu.name}) compute capability unknown "
            f"(torch CUDA not live + name not in heuristic table)"
        )
    cap_str = f"{cap[0]}.{cap[1]}"
    suffix = f" [inferred: {inferred_family}]" if inferred_family else ""
    if cap >= COMPUTE_CAPABILITY_AMPERE:
        return 0, (f"GPU {gpu.index} ({gpu.name}) compute {cap_str}{suffix} - Ampere+; supported")
    return 2, (
        f"GPU {gpu.index} ({gpu.name}) compute {cap_str}{suffix} - "
        f"pre-Ampere; Isaac Sim 5.1 officially requires "
        f"{COMPUTE_CAPABILITY_AMPERE[0]}.{COMPUTE_CAPABILITY_AMPERE[1]}+. "
        f"Headless training may still work via CPU PhysX (~50x slower); "
        f"RTX renderer disabled."
    )


def _check_driver(gpus: Sequence[GpuInfo]) -> tuple[int, str]:
    """Pick driver from first GPU and compare to Isaac minimum."""
    if not gpus:
        return 1, "No GPUs detected via nvidia-smi"
    raw = gpus[0].driver_version
    try:
        major = int(raw.split(".", 1)[0])
    except (ValueError, IndexError):
        return 2, f"Driver version unparseable: {raw}"
    if major >= ISAAC_SIM_MIN_DRIVER_MAJOR:
        return 0, f"NVIDIA driver {raw} (>= {ISAAC_SIM_MIN_DRIVER_MAJOR})"
    return 1, f"NVIDIA driver {raw} too old (need >= {ISAAC_SIM_MIN_DRIVER_MAJOR})"


def _check_packages() -> dict[str, bool]:
    """Return name -> importable for all relevant packages."""
    return {
        name: importlib.util.find_spec(name) is not None
        for name in (*BASE_PACKAGES, *ISAAC_PACKAGES)
    }


def main() -> int:
    print("=" * 64)
    print("armdroid Isaac Sim install / runtime probe")
    print("=" * 64)

    severity = 0  # 0 ok, 1 hard fail, 2 soft warning

    # ---- Python ----
    ok, msg = _check_python()
    print(f"[{'OK' if ok else 'WARN'}] {msg}")
    if not ok:
        severity = max(severity, 2)

    # ---- GPU enumeration: nvidia-smi (rich) + Windows CIM (complete) ----
    nvidia_smi_gpus = _query_nvidia_smi()
    windows_gpus = _query_windows_adapters()
    gpus, merge_warnings = _merge_gpu_sources(nvidia_smi_gpus, windows_gpus)
    if not gpus:
        print("[FAIL] No NVIDIA GPUs detected (nvidia-smi missing AND Windows CIM empty)")
        return 1
    for warning in merge_warnings:
        print(f"[WARN] {warning}")
        severity = max(severity, 2)

    # ---- driver ----
    if nvidia_smi_gpus:
        sev, msg = _check_driver(nvidia_smi_gpus)
        print(f"[{('OK', 'FAIL', 'WARN')[sev]}] {msg}")
        severity = max(severity, sev)
        if sev == 1:
            return 1
    else:
        # nvidia-smi absent - rely on Windows-reported driver string.
        print(
            "[WARN] nvidia-smi unavailable; cannot verify driver "
            "version against the Isaac Sim 5.1 minimum (>= "
            f"{ISAAC_SIM_MIN_DRIVER_MAJOR}). Inspect WDDM strings below."
        )
        severity = max(severity, 2)

    # ---- torch + CUDA runtime ----
    torch_info = _query_torch()
    print()
    if not torch_info.importable:
        print("[FAIL] torch missing - install base [dev] first")
        severity = max(severity, 1)
    else:
        cuda_str = torch_info.cuda_version or "CPU-only"
        print(f"[INFO] torch {torch_info.version} (cuda runtime: {cuda_str})")
        if torch_info.cuda_version is None:
            print(
                "[WARN] torch is a CPU-only build; the [isaac] extra would "
                "pull a CUDA torch and force a reinstall. Detection of "
                "GPU compute capability falls back to nvidia-smi only."
            )
            severity = max(severity, 2)
        elif not torch_info.cuda_runtime_available:
            print(
                "[WARN] torch built with CUDA but cuda runtime not "
                "available (driver mismatch? wrong toolkit?)"
            )
            severity = max(severity, 2)

    # ---- per-GPU compute capability ----
    gpus = _enrich_with_torch(gpus, torch_info)
    print()
    print(f"NVIDIA GPUs detected: {len(gpus)}")
    best_severity = 2  # severity of the *best* GPU; an Ampere+ GPU rescues
    # the verdict even when the system also has a pre-Ampere card.
    for gpu in gpus:
        cap = (
            f"compute {gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
            if gpu.compute_capability
            else "compute capability via name heuristic"
        )
        ram = f"{gpu.memory_mib} MiB" if gpu.memory_mib else "RAM ?"
        print(f"  GPU {gpu.index}: {gpu.name}  |  {ram}  |  {cap}")
        sev, msg = _verdict_for_gpu(gpu)
        print(f"    [{('OK', 'FAIL', 'WARN')[sev]}] {msg}")
        best_severity = min(best_severity, sev)
    # Aggregate severity across the system: if at least one GPU is OK,
    # the system is OK. Otherwise carry the worst (lowest-tier) verdict.
    severity = max(severity, best_severity)

    # ---- packages ----
    print()
    print("Package status:")
    pkg_status = _check_packages()
    width = max(len(name) for name in pkg_status)
    for name in (*BASE_PACKAGES, *ISAAC_PACKAGES):
        present = pkg_status[name]
        category = "base" if name in BASE_PACKAGES else "[isaac]"
        marker = "OK" if present else ("FAIL" if name in BASE_PACKAGES else "MISSING")
        print(f"  [{marker:>7}] {name:<{width}} ({category})")

    isaac_installed = all(pkg_status[name] for name in ISAAC_PACKAGES)
    base_ok = all(pkg_status[name] for name in BASE_PACKAGES)
    if not base_ok:
        print()
        print('[FAIL] base packages missing - run `pip install -e ".[dev]"` first')
        return 1

    print()
    print("=" * 64)
    print("VERDICT")
    print("=" * 64)
    if severity == 0 and isaac_installed:
        print("Hardware + packages OK. The [isaac] extra is installed.")
        print("Next:  ARMDROID_ISAAC_RUN=1 pytest tests/isaac/")
    elif severity == 0 and not isaac_installed:
        print("Hardware OK; [isaac] extra not yet installed. Run:")
        print()
        print('   pip install -e ".[isaac]" \\')
        print("       --extra-index-url https://pypi.nvidia.com")
    elif severity == 2:
        print("Soft warnings present - install may succeed but expect")
        print("degraded behaviour. Specifically:")
        py_cur = sys.version_info[:2]
        if py_cur != ISAAC_LAB_PYTHON:
            print(
                f"  * Python {py_cur[0]}.{py_cur[1]} venv vs Isaac Lab "
                f"{ISAAC_LAB_PYTHON[0]}.{ISAAC_LAB_PYTHON[1]} pin - "
                f"create a 3.{ISAAC_LAB_PYTHON[1]} venv first."
            )
        if torch_info.importable and torch_info.cuda_version is None:
            print(
                "  * torch is CPU-only; [isaac] will reinstall it as a CUDA build (large download)."
            )
        ampere_gpus = [
            gpu for gpu in gpus if _effective_capability(gpu) >= COMPUTE_CAPABILITY_AMPERE
        ]
        if ampere_gpus:
            names = ", ".join(g.name for g in ampere_gpus)
            print(
                f"  * Compatible GPU(s) present: {names}. Set "
                f"CUDA_VISIBLE_DEVICES to pin Isaac Sim to one of these "
                f"once the install completes."
            )
        for gpu in gpus:
            eff_cap = _effective_capability(gpu)
            if eff_cap < COMPUTE_CAPABILITY_AMPERE:
                print(
                    f"  * GPU {gpu.index} ({gpu.name}) compute {eff_cap[0]}."
                    f"{eff_cap[1]} is pre-Ampere; the install will skip this "
                    f"card if a newer one is also present."
                )
        print()
        if ampere_gpus:
            print("Recommended: build a Python 3.11 venv, install CUDA torch,")
            print("then [isaac]; pin to the Ampere+ GPU via CUDA_VISIBLE_DEVICES.")
        else:
            print("Recommended: a Python 3.11 venv on Ampere+ hardware before")
            print("burning the ~9-15 GB install on this box.")
    else:
        print("Hard incompatibility. See [FAIL] lines above.")
    print("=" * 64)
    return severity
    # severity-as-exit-code: 0 ok, 1 hard fail, 2 soft warning


if __name__ == "__main__":
    raise SystemExit(main())
