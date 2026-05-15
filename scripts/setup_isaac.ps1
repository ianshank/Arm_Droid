<#
.SYNOPSIS
    Bootstrap an Isaac Sim 5.1 / Isaac Lab 2.3 development environment for armdroid.

.DESCRIPTION
    Creates a Python 3.11 virtual environment at .venv-isaac (separate from the
    default 3.13 .venv so we don't disturb existing tooling) and installs the
    armdroid [dev,isaac] extras from the NVIDIA pip index. Then runs the
    pure-Python smoke imports defined by PR-B so failures surface before you
    try to launch Kit.

    Isaac Lab 2.3.2.post1 (current pin) requires Python 3.11 and torch>=2.7
    which is built against CUDA 12.8 — that combination ships sm_120 binary
    kernels so Blackwell GPUs (RTX 50-series) work end-to-end. Running it on
    Python 3.12 / 3.13 will fail at import time, hence the dedicated 3.11 venv.

.PARAMETER VenvPath
    Where to create the Isaac venv. Defaults to .\.venv-isaac next to .venv.

.PARAMETER PythonExe
    Path to the Python 3.11 launcher. Defaults to "py -3.11"; overrideable
    if you want to point at a specific install (e.g. an Anaconda env).

.PARAMETER SkipInstall
    Skip the heavy pip install step. Useful for re-running the smoke tests.

.PARAMETER Headless
    Run a single Kit headless smoke ping at the end ("import isaacsim" then
    "from isaaclab.app import AppLauncher; AppLauncher(headless=True)").
    Off by default because it allocates the GPU.

.EXAMPLE
    .\scripts\setup_isaac.ps1
    Standard install: creates .venv-isaac, installs armdroid[dev,isaac].

.EXAMPLE
    .\scripts\setup_isaac.ps1 -SkipInstall -Headless
    Re-run smoke tests against an already-installed venv and ping Kit once.

.NOTES
    PR-B reference: see docs/architecture/ADR/ADR-0005-isaac-sim-backend.md and
    NEXT_STEPS.md in the claude/isaac-runtime-pr-b branch.

    Total install size ~9-10 GB. Make sure C: has at least 20 GB free before
    starting (we measured 111.6 GB free at script-author time).
#>

[CmdletBinding()]
param(
    [string]$VenvPath  = "$PSScriptRoot\..\.venv-isaac",
    [string]$PythonExe = "py -3.11",
    [switch]$SkipInstall,
    [switch]$Headless
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Write-Section($msg) {
    Write-Host ""
    Write-Host "=== $msg ===" -ForegroundColor Cyan
}

function Test-Command($name) {
    return $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

# --- Pre-flight ---------------------------------------------------------------
Write-Section "Pre-flight checks"

# Resolve the project root (script is at <repo>/scripts/setup_isaac.ps1).
# Keep $VenvPath as the bound parameter; assign the resolved location to a
# separate string ($ResolvedVenv) so downstream calls always see a string,
# not the PathInfo that Resolve-Path returns. Type drift under StrictMode
# Latest has bitten us before.
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$ResolvedVenvInfo = Resolve-Path -Path $VenvPath -ErrorAction SilentlyContinue
if ($ResolvedVenvInfo) {
    $ResolvedVenv = $ResolvedVenvInfo.Path
} else {
    $ResolvedVenv = Join-Path $RepoRoot ".venv-isaac"
}
Write-Host "Repo root:  $RepoRoot"
Write-Host "Venv path:  $ResolvedVenv"

# Confirm Python 3.11 is available via the launcher.
$pyVersion = & cmd /c "$PythonExe -V" 2>&1
if ($LASTEXITCODE -ne 0 -or $pyVersion -notmatch '3\.11\.') {
    throw "Python 3.11 not found via '$PythonExe'. Install Python 3.11 (winget install Python.Python.3.11) or pass -PythonExe 'C:\path\to\python311.exe'. Got: $pyVersion"
}
Write-Host "Python:     $pyVersion"

# Confirm GPU + driver. Classify compute capability into the three bands
# that affect the install path:
#   < 7.0   -> pre-Turing (Pascal/Maxwell/Kepler). Isaac Sim 5.x requires
#             sm_75+; recommend the MuJoCo path.
#   12.0+   -> Blackwell (RTX 50-series, B100, GB200). Needs torch >= 2.6
#             with CUDA 12.8 binaries to launch sm_120 kernels; otherwise
#             the install will appear to work until the first GPU op.
#   else    -> Turing/Ampere/Ada/Hopper. No special handling needed.
if (Test-Command nvidia-smi) {
    # nvidia-smi emits one CSV row per GPU; on multi-GPU hosts $gpuLine
    # would be an array and the column index split would be wrong. Force
    # an array and take the first row only.
    $gpuLines = @(& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader)
    $gpuLine  = $gpuLines | Select-Object -First 1
    Write-Host "GPU:        $gpuLine"
    if ($gpuLines.Count -gt 1) {
        Write-Host "            ($($gpuLines.Count) GPUs total; using first for compute-cap classification — set CUDA_VISIBLE_DEVICES if you want a different one)"
    }
    # Guard the parse: nvidia-smi can return 'N/A' for very old or VM-passthrough
    # cards, and StrictMode would crash on the cast.
    $computeStr = ($gpuLine -split ',')[3].Trim()
    $computeNum = 0.0
    if (-not [double]::TryParse($computeStr, [ref]$computeNum)) {
        Write-Warning "Could not parse compute capability '$computeStr' from nvidia-smi. Skipping GPU-class advisories; install will continue."
        $computeNum = 0.0
    }
    if ($computeNum -gt 0.0 -and $computeNum -lt 7.0) {
        Write-Host ""
        Write-Host "  !!! Pre-Turing GPU detected (compute capability $computeStr)." -ForegroundColor Yellow
        Write-Host "  !!! Isaac Sim 5.x requires sm_75 (Turing) or newer." -ForegroundColor Yellow
        Write-Host "  !!! Recommended path on this card:" -ForegroundColor Yellow
        Write-Host "  !!!     .\scripts\setup_mujoco_only.ps1   (MuJoCo path; proven)" -ForegroundColor Yellow
        Write-Host ""

        if ($Headless) {
            Write-Warning "Disabling -Headless Kit smoke on unsupported GPU (most likely to crash)."
            $Headless = $false
        }

        if ($Host.UI.RawUI -and -not $env:CI) {
            $resp = Read-Host "Continue with Isaac install anyway? (y/N)"
            if ($resp -notmatch '^[Yy]') {
                Write-Host "Aborted. Use .\scripts\setup_mujoco_only.ps1 for the proven path." -ForegroundColor Cyan
                exit 2
            }
        }
    }
    elseif ($computeNum -ge 12.0) {
        Write-Host ""
        Write-Host "  Blackwell-class GPU detected (compute capability $computeStr)." -ForegroundColor Cyan
        Write-Host "  Requires torch >= 2.6 with CUDA 12.8 binaries for sm_120 kernels."
        Write-Host "  pyproject pins isaaclab==2.3.2.post1, which pulls torch>=2.7 — good."
        Write-Host "  See docs/GPU_BLACKWELL.md for VRAM-tuned num_envs guidance"
        Write-Host "  (5060 8GB: ~512 envs / 5060 Ti 16GB: ~2048 envs)."
        Write-Host ""
    }
} else {
    throw "nvidia-smi not on PATH. Install/repair the NVIDIA driver before continuing."
}

# Confirm at least 20 GB free on the install drive. The Isaac install itself
# is ~9-10 GB on disk, but pip needs another ~5-10 GB of working space for
# wheel unpacking, build isolation, and caching during the resolution pass.
# 20 GB is the documented requirement in the .NOTES header.
$venvDrive  = (Split-Path -Qualifier $ResolvedVenv).TrimEnd(':') + ':'
$driveInfo  = Get-PSDrive -Name $venvDrive.TrimEnd(':') -ErrorAction Stop
$freeGB     = [math]::Round($driveInfo.Free/1GB, 1)
Write-Host "Free space: $freeGB GB on $venvDrive (need ~20 GB headroom)"
if ($freeGB -lt 20) {
    Write-Warning "Less than 20 GB free on $venvDrive. Isaac install needs ~9-10 GB plus pip working space; consider freeing space first."
}

# --- Venv creation ------------------------------------------------------------
if (-not $SkipInstall) {
    Write-Section "Creating venv at $ResolvedVenv"
    if (Test-Path $ResolvedVenv) {
        Write-Host "Venv already exists; reusing."
    } else {
        & cmd /c "$PythonExe -m venv `"$ResolvedVenv`""
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
    }
}

$VenvPython = Join-Path $ResolvedVenv "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Venv python missing at $VenvPython"
}

# --- Install ------------------------------------------------------------------
if (-not $SkipInstall) {
    Write-Section "Upgrading pip / wheel"
    & $VenvPython -m pip install --upgrade pip wheel setuptools
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

    Write-Section "Installing armdroid[dev,isaac] (this takes 10-30 minutes; ~9-10 GB)"
    Write-Host "Using NVIDIA pip index for isaacsim wheels: https://pypi.nvidia.com"
    Push-Location $RepoRoot
    try {
        & $VenvPython -m pip install -e ".[dev,isaac]" `
            --extra-index-url https://pypi.nvidia.com
        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    } finally {
        Pop-Location
    }
}

# --- Smoke imports (pure Python, no Kit launch) -------------------------------
Write-Section "Smoke imports (pure Python; no GPU touch)"

$smokeScript = @'
import sys, importlib

mods = [
    "armdroid",
    "armdroid.hardware.isaac_sim.gripper",      # pure math; safe
    "armdroid.config.schema.sim_isaac",         # ArmSimIsaacConfig
    "armdroid.config.schema.training",          # RslRlPpoConfig
    "armdroid.environments._tensor_adapter",    # numpy<->torch bridge
]
failed = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  OK   {m}")
    except Exception as e:
        print(f"  FAIL {m}: {type(e).__name__}: {e}")
        failed.append(m)

# Optional: confirm isaaclab is importable WITHOUT booting Kit.
try:
    import isaaclab
    print(f"  OK   isaaclab (version: {getattr(isaaclab,'__version__','?')})")
except Exception as e:
    print(f"  FAIL isaaclab: {type(e).__name__}: {e}")
    failed.append("isaaclab")

try:
    import rsl_rl
    print(f"  OK   rsl_rl (version: {getattr(rsl_rl,'__version__','?')})")
except Exception as e:
    print(f"  FAIL rsl_rl: {type(e).__name__}: {e}")
    failed.append("rsl_rl")

sys.exit(1 if failed else 0)
'@

$smokePath = Join-Path $env:TEMP "armdroid_isaac_smoke.py"
Set-Content -Path $smokePath -Value $smokeScript -Encoding UTF8
& $VenvPython $smokePath
$smokeExit = $LASTEXITCODE

# --- Optional headless Kit ping -----------------------------------------------
if ($Headless -and $smokeExit -eq 0) {
    Write-Section "Headless Kit ping (allocates GPU briefly)"
    $kitScript = @'
import sys
try:
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True)
    print("  OK   AppLauncher booted headless")
    app.app.close()
    sys.exit(0)
except Exception as e:
    print(f"  FAIL AppLauncher: {type(e).__name__}: {e}")
    sys.exit(1)
'@
    $kitPath = Join-Path $env:TEMP "armdroid_isaac_kit_ping.py"
    Set-Content -Path $kitPath -Value $kitScript -Encoding UTF8
    & $VenvPython $kitPath
}

# --- Summary ------------------------------------------------------------------
Write-Section "Summary"
if ($smokeExit -eq 0) {
    Write-Host "Smoke imports passed." -ForegroundColor Green
    Write-Host ""
    Write-Host "Activate the venv and train:"
    Write-Host "    & '$ResolvedVenv\Scripts\Activate.ps1'"
    Write-Host "    python -m armdroid --config config/tower_of_hanoi_isaac.yaml train"
    Write-Host ""
    Write-Host "Run the GPU smoke suite (single-env, requires GPU):"
    Write-Host "    `$env:ARMDROID_ISAAC_RUN = '1'"
    Write-Host "    pytest tests/isaac -n0"
    exit 0
} else {
    Write-Host "Smoke imports FAILED. Inspect output above." -ForegroundColor Red
    Write-Host ""
    Write-Host "If this is a pre-Turing GPU (compute_cap < 7.0), fall back to MuJoCo:" -ForegroundColor Yellow
    Write-Host "    .\scripts\setup_mujoco_only.ps1" -ForegroundColor Yellow
    exit 1
}
