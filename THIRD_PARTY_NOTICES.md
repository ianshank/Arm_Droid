# Third-Party Notices

This file indexes all third-party content vendored into armdroid.
Each entry below points at the directory that holds the vendored content
plus its own per-directory `ATTRIBUTION.md` with full provenance.

## Vendored directories

| Path                                            | Upstream                                      | License        | Pinned SHA |
| ----------------------------------------------- | --------------------------------------------- | -------------- | ---------- |
| `assets/so_arm/so101/`                          | [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) `Simulation/SO101/` | Apache-2.0     | `fda892cba81032c46c40976a48c9ceadbf40a9ca` |
| `src/armdroid/hardware/isaac_sim/articulation.py` | [MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) `src/isaac_so_arm101/robots/trs_so100/so_arm100.py` | BSD-3-Clause   | `e4624dea075b00a36dbc66bebd531d191c92e8cd` |
| `src/armdroid/environments/isaac/tasks/reach/`  | [MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) `src/isaac_so_arm101/tasks/reach/` | BSD-3-Clause   | `e4624dea075b00a36dbc66bebd531d191c92e8cd` |

### Vendored modifications from upstream

* **`articulation.py`** — every PD gain / init pose / solver iteration /
  fix_base / self_collision / root quat / init root pos parametrised on
  ``ArmSimIsaacConfig`` rather than module-level constants. Numeric
  defaults match upstream exactly (verified against
  ``src/isaac_so_arm101/robots/trs_so100/so_arm100.py`` at the pinned
  SHA), so behaviour is identical when ``sim_cfg`` is the default.
* **`tasks/reach/*`** — package imports rewritten via mechanical sed
  rule:
  - `isaac_so_arm101.tasks.reach` → `armdroid.environments.isaac.tasks.reach`
  - `isaac_so_arm101.robots.trs_so100` → `armdroid.hardware.isaac_sim`
  - `isaac_so_arm101.robots` → `armdroid.environments.isaac.robots`
  All other content (gym env registrations, env_cfg dataclasses,
  reward / observation / termination MDP terms) preserved verbatim.
  BSD-3 + The Isaac Lab Project Developers attribution headers
  preserved at the top of every vendored file.

## Refresh policy

Each per-directory `ATTRIBUTION.md` carries the upstream commit SHA,
clone URL, and a refresh procedure. Update SHA pins by re-running the
documented refresh procedure and running `make check` afterwards to
validate that the new vendored content doesn't regress armdroid tests.
