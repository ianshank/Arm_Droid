# GPU note: Blackwell (RTX 50-series, sm_120)

Target hardware for this project includes **NVIDIA RTX 5060 8GB** and **RTX 5060 Ti 16GB** — both Blackwell, compute capability **12.0** (sm_120).

This note covers what makes Blackwell different from the Turing/Ampere/Ada cards Isaac Sim has been validated against for years, and what we did about it in this repo.

## Why Blackwell needs special attention

Blackwell binary kernels (sm_120) were added to NVIDIA's CUDA toolchain in **CUDA 12.8**. PyTorch ships precompiled kernels in its release wheels — and wheels built before torch 2.6 do **not** contain sm_120 binaries.

A torch wheel without sm_120 will:

- **Import fine.** `import torch` succeeds; `torch.cuda.is_available()` returns `True`; `torch.cuda.get_device_name()` returns the right string.
- **Fail the first time a kernel actually runs**, with:
  ```
  CUDA error: no kernel image is available for execution on the device
  ```
  Typical triggers: `torch.zeros(1).cuda()`, `isaaclab.app.AppLauncher`, any training step.

So smoke imports passing **does not** prove the install is good on a 5060/5060 Ti. The kernel check in [Validation](#validation) below is the real gate.

## What we did

[pyproject.toml:47](../pyproject.toml) pins:

```toml
isaac = [
    "isaaclab[all,isaacsim]==2.3.2.post1",
    "rsl-rl-lib>=2.0",
]
```

`isaaclab 2.3.2.post1` requires `torch >= 2.7`, which **does** include sm_120 binaries. The base `torch >= 2.6` floor at [pyproject.toml:11](../pyproject.toml) is the formal floor (2.7 satisfies 2.6); the effective torch on an `[isaac]` install will be 2.7 or newer.

[scripts/setup_isaac.ps1](../scripts/setup_isaac.ps1) and [scripts/install_isaac_offline.ps1](../scripts/install_isaac_offline.ps1) detect Blackwell (`compute_cap >= 12.0`) and print a one-line note pointing at this doc; they do not change install behavior.

## Driver and CUDA matrix

| Component | Minimum for Blackwell | What it gives you |
|-----------|----------------------|-------------------|
| NVIDIA driver | **R570** (Linux) / **572** (Windows) | First driver branch with full sm_120 support |
| CUDA toolkit | **12.8** | First toolkit shipping sm_120 binary kernels |
| PyTorch | **2.6** (recommended **2.7+**) | Binary wheels with sm_120 kernels (NVIDIA cu128 index) |
| Python | **3.11** | Isaac Lab 2.3.2.post1 pins `Requires-Python: ==3.11.*` |

Check the installed driver from PowerShell:

```powershell
$drv = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader).Trim()
Write-Host "Driver: $drv"
if ([version]$drv -lt [version]"572.0") {
    Write-Warning "Driver $drv is below the Blackwell minimum (R572). Update from https://www.nvidia.com/Download/index.aspx."
}
```

## VRAM tuning

Default `num_envs` in Isaac Lab tasks is sized for cards with 24GB+. On the cards in this fleet you'll need to tune down or you'll OOM in the first rollout.

| Card | VRAM | Suggested `num_envs` for Reach | Suggested `num_envs` for Tower of Hanoi |
|------|------|-------------------------------:|----------------------------------------:|
| RTX 5060 8GB | 8 GB | **512** | **256** |
| RTX 5060 Ti 16GB | 16 GB | **2048** | **1024** |

These are starting points — measure peak VRAM with `nvidia-smi --loop-ms=500 --query-gpu=memory.used --format=csv` during a short training run and raise/lower until you have ~1 GB headroom.

The config at [config/tower_of_hanoi_isaac.yaml](../config/tower_of_hanoi_isaac.yaml) is the Blackwell-ready base. If 8GB OOMs persistently, add a `tower_of_hanoi_isaac_5060.yaml` overlay rather than editing the base.

## Validation

Run this immediately after `setup_isaac.ps1`. It is the only sequence that proves end-to-end Blackwell support — `import torch; torch.cuda.is_available()` is not sufficient.

```powershell
& .\.venv-isaac\Scripts\python.exe -c @"
import torch
print('torch                ', torch.__version__)
print('cuda runtime         ', torch.version.cuda)
print('arch list            ', torch.cuda.get_arch_list())
print('device name          ', torch.cuda.get_device_name(0))
print('device capability    ', torch.cuda.get_device_capability(0))

# The actual kernel launch — this is what fails on torch<2.6 / no sm_120.
t = torch.zeros(1, device='cuda')
t += 1
torch.cuda.synchronize()
print('kernel launch        OK', t.item(), t.device)
"@
```

Pass criteria:

- `torch` ≥ 2.6 (typically 2.7.x in practice).
- `arch list` contains `sm_120` (or equivalent — pre-release wheels sometimes call it `compute_120`).
- `device capability` prints `(12, 0)` for the 5060/5060 Ti.
- The final line prints `kernel launch        OK 1.0 cuda:0` with no error.

If the kernel launch errors out, the most common cause is a torch wheel without sm_120 leaked into the env (e.g. via a constraints file or an older cached wheel). Re-run `setup_isaac.ps1` with a fresh `.venv-isaac` and inspect `pip list | findstr torch`.

## Fallback

If the Isaac install fails on the 5060 stack (rare with the `2.3.2.post1` pin, but possible if NVIDIA changes index contents), fall back to MuJoCo:

```powershell
.\scripts\setup_mujoco_only.ps1
python -m armdroid --config config/tower_of_hanoi.yaml train
```

This will not use the GPU for physics (MuJoCo is CPU), but it gets training back online while you debug the Isaac side.
