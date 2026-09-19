# ManiSkill simulation datasets

Four manipulation tasks in LeRobot v3.0 format, each with 40 episodes at 20 FPS. These are the `mixed_30pct` datasets referenced by the training configurations.

| Dataset | Episodes | Frames |
| --- | ---: | ---: |
| [maniskill_stack_cube_mixed_30pct_40ep](maniskill_stack_cube_mixed_30pct_40ep/) | 40 | 5,512 |
| [maniskill_pull_cube_tool_mixed_30pct_40ep](maniskill_pull_cube_tool_mixed_30pct_40ep/) | 40 | 16,214 |
| [maniskill_place_sphere_mixed_30pct_40ep](maniskill_place_sphere_mixed_30pct_40ep/) | 40 | 11,945 |
| [maniskill_lift_peg_upright_mixed_30pct_40ep](maniskill_lift_peg_upright_mixed_30pct_40ep/) | 40 | 11,500 |

## Download

With Git LFS installed:

```sh
git lfs install
git clone https://github.com/anonymousprojectpage2/QLUE.git
cd QLUE
git lfs pull --include="src/maniskill_sim_data/**"
```

The datasets are under `src/maniskill_sim_data/`. Videos are stored in Git LFS; use the commands above to retrieve their full contents.

## Contents

Each dataset retains its `data/`, `meta/`, and `videos/` directories. Read each `meta/info.json` for feature shapes, camera paths, and action/state dimensions. Task instructions are in `meta/tasks.parquet`.

LiftPegUpright-v1 additionally includes the original previews, generation scripts, traces, paired references, and validation records. Its mixture manifest records 28 clean, 4 stop, 4 recovery, and 4 detour episodes. Machine-specific home paths in its provenance have been replaced with `source_workspace/` for anonymous release, and the included checksum manifest has been updated. These source paths are audit references, not runtime dependencies. Data, video, and trace contents are unchanged.

Existing notices in the bundled source files are retained; no new dataset license is assigned by this upload.
