# Third-Party Notices

This file indexes all third-party content vendored into armdroid.
Each entry below points at the directory that holds the vendored content
plus its own per-directory `ATTRIBUTION.md` with full provenance.

## Vendored directories

| Path                         | Upstream                                      | License        | Pinned SHA |
| ---------------------------- | --------------------------------------------- | -------------- | ---------- |
| `assets/so_arm/so101/`       | [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) `Simulation/SO101/` | Apache-2.0     | `fda892cba81032c46c40976a48c9ceadbf40a9ca` |

## Pending — to be added in PR-B

| Path                                            | Upstream                                                                  | License   |
| ----------------------------------------------- | ------------------------------------------------------------------------- | --------- |
| `src/armdroid/environments/isaac/tasks/reach/`  | [MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) `tasks/reach/` | BSD-3-Clause |
| RSL-RL PPO defaults in `config/schema/training.py` | Same — `rsl_rl_ppo_cfg.py` defaults vendored as Pydantic field defaults | BSD-3-Clause |

## Refresh policy

Each per-directory `ATTRIBUTION.md` carries the upstream commit SHA,
clone URL, and a refresh procedure. Update SHA pins by re-running the
documented refresh procedure and running `make check` afterwards to
validate that the new vendored content doesn't regress armdroid tests.
