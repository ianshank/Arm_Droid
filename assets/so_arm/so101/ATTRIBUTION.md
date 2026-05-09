# SO-ARM101 Vendored Assets — Attribution

This directory vendors simulation assets for the SO-ARM101 5DOF arm
(plus its parallel-jaw gripper) from the upstream open-hardware project:

- **Upstream:** [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- **Upstream path:** `Simulation/SO101/`
- **Pinned commit:** `fda892cba81032c46c40976a48c9ceadbf40a9ca`
- **Pin date:** 2026-02-26
- **License:** Apache License 2.0 (see `./LICENSE`)

## What is vendored

| Path                         | Origin (upstream `Simulation/SO101/...`) | Purpose                          |
| ---------------------------- | ---------------------------------------- | -------------------------------- |
| `urdf/so101_new_calib.urdf`  | `so101_new_calib.urdf`                   | Default URDF (recent calibration) |
| `urdf/so101_old_calib.urdf`  | `so101_old_calib.urdf`                   | Pre-2026 calibration variant     |
| `urdf/assets/*.stl`          | `assets/*.stl`                           | Visual + collision meshes (13 STL files) |
| `mjcf/scene.xml`             | `scene.xml`                              | MuJoCo scene wrapper             |
| `mjcf/so101_new_calib.xml`   | `so101_new_calib.xml`                    | MuJoCo MJCF (recent calibration) |
| `mjcf/so101_old_calib.xml`   | `so101_old_calib.xml`                    | MuJoCo MJCF (older calibration)  |
| `mjcf/joints_properties.xml` | `joints_properties.xml`                  | Servo PD-tuning includes         |
| `LICENSE`                    | `LICENSE` (repo root)                    | Apache 2.0 from upstream         |

## What is NOT vendored

- `*.part` (FreeCAD source files) — too large; mechanical sources only.
- `README.md` — out of scope; armdroid carries its own docs.

## Refresh procedure

```bash
git clone --depth 1 https://github.com/TheRobotStudio/SO-ARM100.git /tmp/so-arm100
cd /tmp/so-arm100 && git rev-parse HEAD  # pin this SHA in this file
DEST="<armdroid>/assets/so_arm/so101"
cp Simulation/SO101/so101_new_calib.urdf  "$DEST/urdf/"
cp Simulation/SO101/so101_old_calib.urdf  "$DEST/urdf/"
cp Simulation/SO101/assets/*.stl          "$DEST/urdf/assets/"
cp Simulation/SO101/scene.xml             "$DEST/mjcf/"
cp Simulation/SO101/so101_new_calib.xml   "$DEST/mjcf/"
cp Simulation/SO101/so101_old_calib.xml   "$DEST/mjcf/"
cp Simulation/SO101/joints_properties.xml "$DEST/mjcf/"
cp LICENSE                                "$DEST/LICENSE"
```

After refresh, run `make check` to validate that no URDF mesh references
broke and the existing simulation tests still pass.

## Mesh path convention

Upstream URDFs reference meshes with `<mesh filename="assets/<file>.stl"/>`.
We preserve that path layout (`urdf/assets/*.stl`) so URDF loaders resolve
meshes from `urdf/` without modification.
