<#
.SYNOPSIS
    Install armdroid[dev,isaac] (or just [dev]) entirely offline from a
    wheel cache produced by download_isaac_wheels.ps1.

.DESCRIPTION
    Creates the venv if missing, then `pip install --no-index
    --find-links <cache> -e ".[dev,isaac]"`. No network calls. Safe to
    run in Safe Mode or any offline boot.

    Detects pre-Turing GPUs (compute_cap < 7.0) and steers you toward the
    [dev]-only install + MuJoCo training path, since Isaac Sim 5.x requires
    sm_75+. Blackwell GPUs (compute_cap >= 12.0, RTX 50-series) install the
    full [isaac] extra normally — see docs/GPU_BLACKWELL.md.

.PARAMETER CacheDir
    Wheel cache directory. Default: <repo>\vendor\wheels.

.PARAMETER VenvPath
    Where to create / reuse the venv. Default: .venv-isaac for the full
    [dev,isaac] install, .venv for [dev]-only.

.PARAMETER PythonExe
    Python launcher. Default: "py -3.11".

.PARAMETER DevOnly
    Skip [isaac] extras even if wheels are present. Targets the .venv
    venv by default. Use this on pre-Turing GPUs (Pascal and older).

.EXAMPLE
    .\scripts\install_isaac_offline.ps1                 # full Isaac install
    .\scripts\install_isaac_offline.ps1 -DevOnly        # MuJoCo path only
#>

[CmdletBinding()]
param(
    [string]$CacheDir  = "$PSScriptRoot\..\vendor\wheels",
    [string]$VenvPath  = "",
    [string]$PythonExe = "py -3.11",
    [switch]$DevOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Section($msg) {
    Write-Host ""
    Write-Host "=== $msg ===" -ForegroundColor Cyan
}

$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
if (-not $VenvPath) {
    $VenvPath = if ($DevOnly) { Join-Path $RepoRoot ".venv" } else { Join-Path $RepoRoot ".venv-isaac" }
}

Section "Pre-flight"
Write-Host "Repo root: $RepoRoot"
Write-Host "Cache dir: $CacheDir"
Write-Host "Venv:      $VenvPath"

if (-not (Test-Path $CacheDir)) {
    throw "Wheel cache not found at $CacheDir. Run download_isaac_wheels.ps1 from an online machine first."
}
$cachedCount = (Get-ChildItem $CacheDir -Recurse -File).Count
if ($cachedCount -eq 0) {
    throw "Cache dir $CacheDir is empty. Run download_isaac_wheels.ps1 first."
}
Write-Host "Wheels:    $cachedCount cached files"

# Classify GPU. Pre-Turing (compute < 7.0) steers toward -DevOnly;
# Blackwell (>= 12.0) gets a heads-up about the torch>=2.6 requirement.
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpuLine = & nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
    Write-Host "GPU:       $gpuLine"
    $compute = ($gpuLine -split ',')[1].Trim()
    $computeNum = [double]$compute
    if ($computeNum -lt 7.0 -and -not $DevOnly) {
        Write-Warning "Pre-Turing GPU (compute $compute). Isaac Sim 5.x requires sm_75+."
        Write-Warning "Recommend: re-run with -DevOnly for the MuJoCo path. Continuing anyway since you didn't pass -DevOnly."
    }
    elseif ($computeNum -ge 12.0 -and -not $DevOnly) {
        Write-Host "Blackwell GPU (compute $compute) — needs torch>=2.6 / CUDA 12.8 in the cache." -ForegroundColor Cyan
        Write-Host "See docs/GPU_BLACKWELL.md for the driver/CUDA matrix." -ForegroundColor Cyan
    }
}

# Venv.
if (-not (Test-Path $VenvPath)) {
    Section "Creating venv at $VenvPath"
    & cmd /c "$PythonExe -m venv `"$VenvPath`""
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
} else {
    Write-Host "Venv exists; reusing."
}
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) { throw "Venv python missing at $VenvPython" }

# Pip from cache only — pip itself can be in the wheel cache too.
Section "Upgrading pip / wheel / setuptools (offline)"
& $VenvPython -m pip install --no-index --find-links "$CacheDir" --upgrade pip wheel setuptools
# This may exit nonzero if those wheels weren't cached — not fatal, continue.

Section "Installing armdroid (offline)"
$extras = if ($DevOnly) { "[dev]" } else { "[dev,isaac]" }
Push-Location $RepoRoot
try {
    & $VenvPython -m pip install --no-index --find-links "$CacheDir" -e ".$extras"
    if ($LASTEXITCODE -ne 0) {
        throw "Offline install failed. Check that all required wheels are in $CacheDir."
    }
} finally {
    Pop-Location
}

Section "Smoke imports"
$smoke = if ($DevOnly) {
@'
import armdroid; print("OK armdroid")
'@
} else {
@'
import armdroid; import isaaclab; import rsl_rl; print("OK armdroid + isaaclab + rsl_rl")
'@
}
$smokePath = Join-Path $env:TEMP "armdroid_offline_smoke.py"
Set-Content -Path $smokePath -Value $smoke -Encoding UTF8
& $VenvPython $smokePath
$rc = $LASTEXITCODE

Section "Summary"
if ($rc -eq 0) {
    Write-Host "Offline install OK." -ForegroundColor Green
    Write-Host ""
    Write-Host "Activate:"
    Write-Host "    & '$VenvPath\Scripts\Activate.ps1'"
    if ($DevOnly) {
        Write-Host "Train (MuJoCo):"
        Write-Host "    python -m armdroid --config config/tower_of_hanoi.yaml train"
    } else {
        Write-Host "Train (Isaac):"
        Write-Host "    python -m armdroid --config config/tower_of_hanoi_isaac.yaml train"
    }
    exit 0
} else {
    Write-Host "Smoke FAILED. Inspect output above." -ForegroundColor Red
    exit 1
}
