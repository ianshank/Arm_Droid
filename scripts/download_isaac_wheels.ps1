<#
.SYNOPSIS
    Pre-download every Python wheel needed by `armdroid[dev,isaac]` to a
    local cache folder. Run this in NORMAL mode (with WiFi). Install
    offline later via install_isaac_offline.ps1.

.DESCRIPTION
    Uses `pip download` with the NVIDIA pip index to grab wheels for:
      - armdroid base + [dev] + [isaac] dependency closure
      - isaaclab[all,isaacsim]==2.3.2.post1
      - rsl-rl-lib>=2.0
      - torch / numpy / pydantic / ... transitively
    Total size: ~10-15 GB (mostly isaaclab + torch + CUDA libs).

    Defaults target the install machine (this PC) -- same OS, same CPU
    arch, same Python version. If you're downloading on a different
    machine to copy over, pass -CrossPlatform to lock to win_amd64
    + Python 3.11.

.PARAMETER CacheDir
    Where wheels go. Default: <repo>\vendor\wheels.

.PARAMETER PythonExe
    Python launcher. Default: "py -3.11" (matches the Isaac Lab venv).

.PARAMETER CrossPlatform
    Force win_amd64 + Python 3.11 download (use when downloading on a
    different OS / arch / Python version than the install target).

.PARAMETER Mujoco
    Also download wheels for the MuJoCo-only path (tiny, ~500 MB) so you
    can fall back to it offline.

.EXAMPLE
    # Normal case: same machine, online -> later offline.
    .\scripts\download_isaac_wheels.ps1

.EXAMPLE
    # Downloading on a colleague's Mac to USB-stick to your Windows box.
    .\scripts\download_isaac_wheels.ps1 -CrossPlatform

.NOTES
    The cache folder is a flat directory of .whl + .tar.gz files. Copy
    the whole directory to the target machine (USB stick / network
    drive / OneDrive sync after one online boot), then run
    install_isaac_offline.ps1 there.
#>

[CmdletBinding()]
param(
    [string]$CacheDir      = "$PSScriptRoot\..\vendor\wheels",
    [string]$PythonExe     = "py -3.11",
    [switch]$CrossPlatform,
    [switch]$Mujoco
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Section($msg) {
    Write-Host ""
    Write-Host "=== $msg ===" -ForegroundColor Cyan
}

$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$CacheDir = if (Test-Path $CacheDir) { (Resolve-Path $CacheDir).Path } else { $CacheDir }

Section "Setup"
Write-Host "Repo root: $RepoRoot"
Write-Host "Cache dir: $CacheDir"
$pyVer = & cmd /c "$PythonExe -V" 2>&1
Write-Host "Python:    $pyVer"

if (-not (Test-Path $CacheDir)) {
    New-Item -ItemType Directory -Path $CacheDir | Out-Null
}

# Pin setuptools < 70 in build-isolation envs so old sdists like flatdict 4.0.1
# (transitive dep of isaaclab) can `import pkg_resources` -- setuptools 80
# removed it. Applies to all subsequent pip download calls in this process.
$constraintsFile = Join-Path $env:TEMP "armdroid_pip_constraints.txt"
@(
    "setuptools<70"
    "wheel<0.42"
) | Set-Content -Path $constraintsFile -Encoding ASCII
$env:PIP_CONSTRAINT = $constraintsFile
Write-Host "Build-env pin: setuptools<70  (PIP_CONSTRAINT=$constraintsFile)"

# Cross-platform flags (lock to win_amd64 + Python 3.11 ABI).
$xplat = @()
if ($CrossPlatform) {
    $xplat = @(
        "--platform", "win_amd64",
        "--python-version", "3.11",
        "--implementation", "cp",
        "--abi", "cp311",
        "--only-binary=:all:"
    )
    Write-Host "Cross-platform mode: locking win_amd64 + cp311."
}

Section "Downloading build backend (hatchling, pip, wheel, setuptools, editables)"
# These are needed to install armdroid in editable mode offline -- pip needs
# to be able to invoke the build backend even for `pip install -e .` because
# pyproject.toml's [build-system] requires hatchling.
Push-Location $RepoRoot
try {
    & cmd /c "$PythonExe -m pip download -d `"$CacheDir`" pip wheel setuptools hatchling editables $($xplat -join ' ')"
    if ($LASTEXITCODE -ne 0) { throw "pip download (build backend) failed" }
} finally {
    Pop-Location
}

Section "Downloading armdroid[dev] dependency closure"
# Note: NO -e here. `pip download` does not support editable mode (it makes
# no sense -- there's nothing to download for a local editable install).
# Instead we download the dependency closure of `.[dev]`. The local armdroid
# package itself doesn't need to be in the cache because the offline install
# uses `pip install -e .` against the local source tree.
Push-Location $RepoRoot
try {
    & cmd /c "$PythonExe -m pip download -d `"$CacheDir`" `".[dev]`" $($xplat -join ' ')"
    if ($LASTEXITCODE -ne 0) { throw "pip download (dev) failed" }
} finally {
    Pop-Location
}

if ($Mujoco) {
    Section "MuJoCo-only deps already covered by [dev]; skipping separate pass."
}

Section "Downloading [isaac] extras (isaaclab + rsl-rl + torch closure)"
Write-Host "Using NVIDIA pip index for isaacsim wheels: https://pypi.nvidia.com"
Push-Location $RepoRoot
try {
    & cmd /c "$PythonExe -m pip download -d `"$CacheDir`" `"isaaclab[all,isaacsim]==2.3.2.post1`" `"rsl-rl-lib>=2.0`" --extra-index-url https://pypi.nvidia.com $($xplat -join ' ')"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "isaaclab download failed. Common causes:"
        Write-Warning "  1. Pascal-class GPU detected and you'd rather use the MuJoCo path. That's fine -- the [dev] download above is enough."
        Write-Warning "  2. Network hiccup. Re-run; pip download is resumable for already-fetched wheels."
        Write-Warning "  3. NVIDIA index temporarily down. Retry in a few minutes."
        throw "pip download (isaac) failed"
    }
} finally {
    Pop-Location
}

Section "Summary"
$sizeMB = [math]::Round((Get-ChildItem $CacheDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
$count  = (Get-ChildItem $CacheDir -Recurse -File).Count
Write-Host "Wheels cached: $count files, $sizeMB MB" -ForegroundColor Green
Write-Host ""
Write-Host "To install offline (later, in Safe Mode or any offline boot):"
Write-Host "    .\scripts\install_isaac_offline.ps1"
Write-Host ""
Write-Host "Or manually:"
Write-Host "    py -3.11 -m venv .venv-isaac"
Write-Host "    & .\.venv-isaac\Scripts\Activate.ps1"
Write-Host "    pip install --no-index --find-links `"$CacheDir`" -e `".[dev,isaac]`""
