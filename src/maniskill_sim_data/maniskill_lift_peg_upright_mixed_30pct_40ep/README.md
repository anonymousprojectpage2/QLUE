# LiftPegUpright-v1 mixed30 / 40 episodes

Native LeRobot v3 dataset. 28 clean, 4 stop, 4 recovery, 4 detour episodes.
Mixed percentage is by episode count, not frame count.

- Robot: panda_wristcam; controller: pd_joint_pos; action: 8D; qpos state: 9D.
- FPS: 20; frames: 11500; cameras: base_camera and hand_camera, 128x128 RGB.
- Task instruction: Stand the peg upright on the table.
- Environment seeds and source conditions: provenance/mixture_manifest.json.
- Final success is sustained for at least 10 frames; no retained trajectory is truncated.
- Stop: two 20-frame intervals, <=5mm measured drift.
- Recovery: measured failed grasp/insertion, retreat, retry, final success.
- Detour: two verified waypoints, maintained grasp, TCP path >=1.25x paired clean.
- Raw actions were replayed independently and verified for final success and behavior semantics.
- RGB is rendered from recorded simulator states. Action replay is not guaranteed bitwise identical;
  per-episode deviations and success-flag comparisons are in provenance/replay_validation.json.
- Dataset files are standalone. Raw/replay paths in provenance are audit references, not runtime dependencies.
- Source code, exact generator snapshots, dependency versions and excluded attempts are included in provenance/.
- No policy training or Hub upload was performed as part of dataset creation.

For pi0: map observation.images.base_camera to observation.images.base_0_rgb and
observation.images.hand_camera to observation.images.left_wrist_0_rgb, preserving base-then-wrist order.
Use a loader supporting 8D joint-position actions; do not use a 4D Cartesian-action configuration.
