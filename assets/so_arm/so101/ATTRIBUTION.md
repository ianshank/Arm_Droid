# SO-ARM101 Vendored Assets — Attribution

This directory vendors simulation assets for the SO-ARM101 5-DoF arm
(plus its parallel-jaw gripper) from the upstream open-hardware project:

- **Upstream:** [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- **Upstream path:** `Simulation/SO101/`
- **Pinned commit:** `fda892cba81032c46c40976a48c9ceadbf40a9ca`
- **Pin date:** 2026-02-26
- **License:** Apache License 2.0 (see `./LICENSE`)

## Layout — flat, matches upstream verbatim

The directory mirrors `Simulation/SO101/` from upstream so vendored URDF
+ MJCF mesh references resolve identically without modifying any
vendored XML:

| Path                         | Origin (upstream `Simulation/SO101/...`) | Purpose                          |
| ---------------------------- | ---------------------------------------- | -------------------------------- |
| `so101_new_calib.urdf`       | `so101_new_calib.urdf`                   | Default URDF (recent calibration) |
| `so101_old_calib.urdf`       | `so101_old_calib.urdf`                   | Pre-2026 calibration variant     |
| `so101_new_calib.xml`        | `so101_new_calib.xml`                    | MuJoCo MJCF (recent calibration) |
| `so101_old_calib.xml`        | `so101_old_calib.xml`                    | MuJoCo MJCF (older calibration)  |
| `scene.xml`                  | `scene.xml`                              | MuJoCo scene wrapper             |
| `joints_properties.xml`      | `joints_properties.xml`                  | Servo PD-tuning includes         |
| `assets/*.stl`               | `assets/*.stl`                           | Visual + collision meshes (13 STL files) |
| `LICENSE`                    | `LICENSE` (repo root)                    | Apache 2.0 from upstream         |

## Mesh path convention

Both the URDF and the MJCF reference meshes via the same path syntax:

* URDF: `<mesh filename="assets/<file>.stl"/>` — relative to the URDF file's directory.
* MJCF: `<compiler meshdir="assets" .../>` — same.

The flat layout is the *only* layout that lets a single `assets/*.stl`
directory serve both consumers without modifying upstream XML or
duplicating 16 MB of meshes. Earlier drafts of this PR placed URDFs
under `urdf/` and MJCFs under `mjcf/` with STLs only at `urdf/assets/`;
that broke MuJoCo loading because `mjcf/`'s `meshdir="assets"` resolved
to `mjcf/assets/` (which didn't exist). See PR #10 review feedback.

## What is NOT vendored

- `*.part` (FreeCAD source files) — too large; mechanical sources only.
- `README.md` — out of scope; armdroid carries its own docs.
- `usd/*.usd` — generated build artefact (Isaac Sim URDF importer output);
  excluded from git via `.gitignore`.

## Refresh procedure

```bash
git clone --depth 1 https://github.com/TheRobotStudio/SO-ARM100.git /tmp/so-arm100
cd /tmp/so-arm100 && git rev-parse HEAD  # pin this SHA in this file
DEST="<armdroid>/assets/so_arm/so101"
cp Simulation/SO101/so101_new_calib.urdf  "$DEST/"
cp Simulation/SO101/so101_old_calib.urdf  "$DEST/"
cp Simulation/SO101/scene.xml             "$DEST/"
cp Simulation/SO101/so101_new_calib.xml   "$DEST/"
cp Simulation/SO101/so101_old_calib.xml   "$DEST/"
cp Simulation/SO101/joints_properties.xml "$DEST/"
cp Simulation/SO101/assets/*.stl          "$DEST/assets/"
cp LICENSE                                "$DEST/LICENSE"
```

After refresh, run `make check` to validate that no URDF mesh references
broke and the existing simulation tests still pass.
