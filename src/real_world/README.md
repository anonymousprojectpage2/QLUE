# Real-world demonstration datasets

Three SO-101 manipulation datasets in LeRobot v3.0 format, each with 40 episodes and a 30% suboptimal mixture. Videos contain front and top camera views at 30 FPS.

| Task directory | Episodes | Frames |
| --- | ---: | ---: |
| [office_task_mixed_30pct](office_task_mixed_30pct/) | 40 | 22,522 |
| [multi_pick_and_place_mixed_30pct](multi_pick_and_place_mixed_30pct/) | 40 | 21,389 |
| [stack_cube_mixed_30pct](stack_cube_mixed_30pct/) | 40 | 14,742 |

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

Each task contains `data/` (frame-level Parquet data), `videos/` (camera recordings), and `meta/` (dataset information, episode/task tables, statistics, and mixture provenance). See `meta/info.json` for feature names, shapes, paths, and the `train` split (`0:40`). Mixture JSON files preserve source episode indices and mixture settings; machine-specific source paths have been replaced with relative `source_datasets/` paths for anonymous release. Personal dataset visualization links have been removed from the copied dataset cards. Data and video contents are unchanged.

## License

The office-task and multi-pick-and-place dataset cards retain their Apache-2.0 license declarations. The stack-cube source did not supply a dataset card or license declaration; this release does not assign it a new license.
