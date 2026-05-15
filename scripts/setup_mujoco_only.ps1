<#
.SYNOPSIS
    Install the armdroid runtime WITHOUT Isaac Sim -- the supported path
    for pre-Turing GPUs (Pascal and older) and any machine where the
    [isaac] extra is overkill or unwanted.

.DESCRIPTION
    Reuses the existing .venv if one is present, otherwise creates a new
    one on the Python version you point at via -PythonExe (default
    "py -3.11" — matches the project's CI and the Isaac venv). Installs
    `armdroid[dev]` only — no isaaclab, no NVIDIA pip index. Then runs
    the standard non-isaac / non-gpu pytest gate. Total install <500 MB.

    Use this on cards where Isaac Sim 5.x will not run (sm < 7.5) or
    where the user does not need photoreal rendering / Isaac Lab
    training. Blackwell GPUs (RTX 50-series) do NOT need this path —
    use setup_isaac.ps1 instead.

.EXAMPLE
    .\scripts\setup_mujoco_only.ps1
#>

[CmdletBinding()]
param(
    [string]$VenvPath  = "$PSScriptRoot\..\.venv",
    [string]$PythonExe = "py -3.11",
    [switch]$SkipInstall,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Section($msg) {
    Write-Host ""
    Write-Host "=== $msg ===" -ForegroundColor Cyan
}

$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$ResolvedVenv = Resolve-Path -Path $VenvPath -ErrorAction SilentlyContinue
if (-not $ResolvedVenv) { $ResolvedVenv = Join-Path $RepoRoot ".venv" }

Section "Pre-flight"
Write-Host "Repo root: $RepoRoot"
Write-Host "Venv:      $ResolvedVenv"
$pyVer = & cmd /c "$PythonExe -V" 2>&1
Write-Host "Python:    $pyVer"

if (-not $SkipInstall) {
    if (-not (Test-Path $ResolvedVenv)) {
        Section "Creating venv"
        & cmd /c "$PythonExe -m venv `"$ResolvedVenv`""
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
    } else {
        Write-Host "Venv exists; reusing."
    }
}

$VenvPython = Join-Path $ResolvedVenv "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) { throw "Venv python missing at $VenvPython" }

if (-not $SkipInstall) {
    Section "Upgrading pip"
    & $VenvPython -m pip install --upgrade pip wheel setuptools
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

    Section "Installing armdroid[dev] (no Isaac)"
    Push-Location $RepoRoot
    try {
        & $VenvPython -m pip install -e ".[dev]"
        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    } finally {
        Pop-Location
    }
}

if (-not $SkipTests) {
    Section "Running default test gate (no isaac, no gpu markers)"
    Push-Location $RepoRoot
    try {
        & $VenvPython -m pytest -m "not isaac and not gpu" -x --no-header -q
        $rc = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($rc -ne 0) {
        Write-Warning "Tests failed (exit $rc). Inspect output before training."
    } else {
        Write-Host "Tests green." -ForegroundColor Green
    }
}

Section "Next steps"
Write-Host "Activate the venv:"
Write-Host "    & '$ResolvedVenv\Scripts\Activate.ps1'"
Write-Host ""
Write-Host "Train Tower of Hanoi (MuJoCo + SAC+HER):"
Write-Host "    python -m armdroid --config config/tower_of_hanoi.yaml train"
Write-Host ""
Write-Host "Deploy to the ESP32 arm afterward (see firmware/arm_esp32/PROTOCOL.md)."
