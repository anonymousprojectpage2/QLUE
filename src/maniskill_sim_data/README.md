# ManiSkill simulation datasets

Four manipulation tasks: mixed datasets in LeRobot v3.0 format and pre-mixture source pools in ManiSkill HDF5 format. The mixed datasets each contain 40 episodes at 20 FPS and are referenced by the training configurations.

## Mixed datasets

| Dataset | Episodes | Frames |
| --- | ---: | ---: |
| [maniskill_stack_cube_mixed_30pct_40ep](maniskill_stack_cube_mixed_30pct_40ep/) | 40 | 5,512 |
| [maniskill_pull_cube_tool_mixed_30pct_40ep](maniskill_pull_cube_tool_mixed_30pct_40ep/) | 40 | 16,214 |
| [maniskill_place_sphere_mixed_30pct_40ep](maniskill_place_sphere_mixed_30pct_40ep/) | 40 | 11,945 |
| [maniskill_lift_peg_upright_mixed_30pct_40ep](maniskill_lift_peg_upright_mixed_30pct_40ep/) | 40 | 11,500 |

## Pre-mixture source datasets

| Dataset | Episodes | Action steps | Format |
| --- | ---: | ---: | --- |
| [maniskill_stack_cube_clean_40ep](maniskill_stack_cube_clean_40ep/) | 40 | 4,572 | RGB HDF5 |
| [maniskill_stack_cube_stop_40ep](maniskill_stack_cube_stop_40ep/) | 40 | 6,176 | RGB HDF5 |
| [maniskill_stack_cube_detour_40ep](maniskill_stack_cube_detour_40ep/) | 40 | 6,701 | RGB HDF5 |
| [maniskill_stack_cube_recovery_40ep](maniskill_stack_cube_recovery_40ep/) | 40 | 10,535 | RGB HDF5 |
| [maniskill_pull_cube_tool_clean_100ep](maniskill_pull_cube_tool_clean_100ep/) | 100 | 38,342 | RGB HDF5 |
| [maniskill_pull_cube_tool_stop_30ep](maniskill_pull_cube_tool_stop_30ep/) | 30 | 12,149 | RGB HDF5 |
| [maniskill_pull_cube_tool_detour_30ep](maniskill_pull_cube_tool_detour_30ep/) | 30 | 13,081 | RGB HDF5 |
| [maniskill_pull_cube_tool_recovery_30ep](maniskill_pull_cube_tool_recovery_30ep/) | 30 | 15,363 | RGB HDF5 |
| [maniskill_place_sphere_clean_40ep](maniskill_place_sphere_clean_40ep/) | 40 | 10,720 | RGB HDF5 |
| [maniskill_place_sphere_stop_30ep](maniskill_place_sphere_stop_30ep/) | 30 | 9,227 | RGB HDF5 |
| [maniskill_place_sphere_detour_30ep](maniskill_place_sphere_detour_30ep/) | 30 | 9,948 | RGB HDF5 |
| [maniskill_place_sphere_recovery_30ep](maniskill_place_sphere_recovery_30ep/) | 30 | 14,341 | RGB HDF5 |
| [maniskill_lift_peg_upright_clean_67ep](maniskill_lift_peg_upright_clean_67ep/) | 67 | 17,216 | Raw HDF5 + NPZ |
| [maniskill_lift_peg_upright_stop_52ep](maniskill_lift_peg_upright_stop_52ep/) | 52 | 15,059 | Raw HDF5 + NPZ |
| [maniskill_lift_peg_upright_detour_30ep](maniskill_lift_peg_upright_detour_30ep/) | 30 | 11,204 | Raw HDF5 + NPZ |
| [maniskill_lift_peg_upright_recovery_61ep](maniskill_lift_peg_upright_recovery_61ep/) | 61 | 22,642 | Raw HDF5 + NPZ |

StackCube `normal` is named `clean` here. PullCubeTool clean retains the full 100-episode source pool. LiftPegUpright retains all 67 clean, 52 stop, 30 detour, and 61 recovery source episodes; it is not restricted to the 28/4/4/4 mixed-dataset selection.

For StackCube, PullCubeTool, and PlaceSphere, each folder contains a standalone `trajectory.h5` and its matching JSON. Link-only source HDF5 files are omitted. Original HDF5 trajectory IDs and RGB replay episode ordering are preserved; StackCube ordered-link wrappers are not included. LiftPegUpright retains per-seed simulator trajectories and traces without RGB observations. These source pools require conversion before use with a LeRobot-only loader.

After installing `h5py`, inspect a replay file with:

```python
import h5py
with h5py.File("trajectory.h5", "r") as data:
    print(list(data.keys()))
```

## Download

With Git LFS installed:

```sh
git lfs install
git clone https://github.com/anonymousprojectpage2/QLUE.git
cd QLUE
git lfs pull --include="src/maniskill_sim_data/**"
```

The datasets are under `src/maniskill_sim_data/`. Videos and HDF5 files are stored in Git LFS; use the commands above to retrieve their full contents.

## Contents

Each mixed dataset retains its `data/`, `meta/`, and `videos/` directories. Read each `meta/info.json` for feature shapes, camera paths, and action/state dimensions. Task instructions are in `meta/tasks.parquet`.

LiftPegUpright-v1 additionally includes the original previews, generation scripts, traces, paired references, and validation records. Its mixture manifest records 28 clean, 4 stop, 4 recovery, and 4 detour episodes. Machine-specific home paths in its provenance have been replaced with `source_workspace/` for anonymous release, and the included checksum manifest has been updated. These source paths are audit references, not runtime dependencies. Data, video, and trace contents are unchanged.

Existing notices in the bundled source files are retained; no new dataset license is assigned by this upload.
