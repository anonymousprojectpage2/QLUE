# Real-world demonstration datasets

SO-101 manipulation datasets in LeRobot v3.0 format: three mixed datasets and eight behavior-specific datasets. Videos contain front and top camera views at 30 FPS.

## Mixed datasets

| Task directory | Episodes | Frames |
| --- | ---: | ---: |
| [office_task_mixed_30pct](office_task_mixed_30pct/) | 40 | 22,522 |
| [multi_pick_and_place_mixed_30pct](multi_pick_and_place_mixed_30pct/) | 40 | 21,389 |
| [stack_cube_mixed_30pct](stack_cube_mixed_30pct/) | 40 | 14,742 |

## Behavior-specific datasets

Directories follow `<task>_<behavior>_<episode_count>ep`.

| Task directory | Episodes | Frames |
| --- | ---: | ---: |
| [stack_cube_stop_20ep](stack_cube_stop_20ep/) | 20 | 9,051 |
| [stack_cube_detour_30ep](stack_cube_detour_30ep/) | 30 | 13,777 |
| [office_task_recovery_30ep](office_task_recovery_30ep/) | 30 | 20,275 |
| [office_task_stop_30ep](office_task_stop_30ep/) | 30 | 18,765 |
| [office_task_detour_30ep](office_task_detour_30ep/) | 30 | 18,702 |
| [multi_pick_and_place_detour_30ep](multi_pick_and_place_detour_30ep/) | 30 | 16,876 |
| [multi_pick_and_place_recovery_30ep](multi_pick_and_place_recovery_30ep/) | 30 | 22,047 |
| [multi_pick_and_place_stop_30ep](multi_pick_and_place_stop_30ep/) | 30 | 17,185 |

## Download

Install Git LFS, then run:

```sh
git lfs install
git clone https://github.com/anonymousprojectpage2/QLUE.git
cd QLUE
git lfs pull --include="src/real_world/**"
```

The actual datasets are under `src/real_world/`. MP4 files are stored in GitHub Git LFS; the commands above download their full contents. Use these commands instead of relying on the source ZIP archive to include the videos.

## Structure

Each task contains `data/` (frame-level Parquet data), `videos/` (camera recordings), and `meta/` (dataset information, episode/task tables, statistics, and available mixture or merge provenance). See each `meta/info.json` for feature names, shapes, paths, and the `train` split. Mixture JSON files preserve source episode indices and mixture settings; machine-specific source paths have been replaced with relative `source_datasets/` paths for anonymous release. Personal dataset visualization links have been removed from the copied dataset cards. Data and video contents are unchanged. The stack-cube stop dataset also retains the source's `.pre_task_rename.bak` files; these are backups, not additional episodes.

## License

The office-task and multi-pick-and-place dataset cards retain their Apache-2.0 license declarations. The stack-cube sources did not supply dataset cards or license declarations; this release does not assign them new licenses.
