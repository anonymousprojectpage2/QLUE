"""Attach source, replay audits, hashes and usage notes to a validated dataset."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import av
import h5py
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--root',required=True)
p.add_argument('--pool',required=True)
p.add_argument('--replayed',required=True)
args=p.parse_args()
root=Path(args.root).resolve();pool=Path(args.pool).resolve();replayed=Path(args.replayed).resolve()
prov=root/'provenance'
validation=json.loads((prov/'validation.json').read_text());assert validation['valid'] and validation['episodes']==40
manifest=json.loads((prov/'mixture_manifest.json').read_text())
audits=[]; raw_manifests=[]; diagnostics=[]
(prov/'traces').mkdir(exist_ok=True)
(prov/'paired_clean_references').mkdir(exist_ok=True)
for ep in manifest['episodes']:
    audit=json.loads((replayed/ep['condition']/f"seed_{ep['seed']:04d}/replay_validation.json").read_text())
    assert audit['action_replay_final_success'] and audit['action_replay_behavior_valid'] and audit['state_replay_success_flags_equal']
    audits.append(audit)
    raw=json.loads((Path(ep['source_dir'])/'manifest.json').read_text());raw_manifests.append(raw)
    with np.load(Path(ep['source_dir'])/'trace.npz') as trace:
        assert trace['success'][-10:].all() and not trace['truncated'].any()
        diagnostic={'episode_index':ep['episode_index'],
                    'last10_max_linear_speed_m_s':float(np.linalg.norm(trace['linear_velocity'][-10:],axis=1).max()),
                    'last10_max_angular_speed_rad_s':float(np.linalg.norm(trace['angular_velocity'][-10:],axis=1).max())}
        if ep['condition']=='detour':
            start=raw['events']['waypoint_0'][0];end=raw['events']['waypoint_1'][1]
            returned=np.flatnonzero(np.linalg.norm(trace['tcp'][end:,:3]-trace['tcp'][start,:3],axis=1)<0.015)
            assert len(returned)
            return_frame=end+int(returned[0]);assert trace['grasp'][start:return_frame+1].all()
            diagnostic['detour_grasp_maintained_through_return']=True
        diagnostics.append(diagnostic)
    shutil.copy2(Path(ep['source_dir'])/'trace.npz',prov/'traces'/f"episode_{ep['episode_index']:03d}.npz")
    if ep['condition']!='clean':
        reference=Path(ep['clean_reference_dir'])
        shutil.copy2(reference/'trace.npz',prov/'paired_clean_references'/f"seed_{ep['seed']:04d}.npz")
        shutil.copy2(reference/'manifest.json',prov/'paired_clean_references'/f"seed_{ep['seed']:04d}.json")
(prov/'replay_validation.json').write_text(json.dumps(audits,indent=2)+'\n')
(prov/'source_episode_manifests.json').write_text(json.dumps(raw_manifests,indent=2)+'\n')
(prov/'additional_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
failures=[]
for pth in sorted((pool/'failed').glob('*.json')):
    failures.append(json.loads(pth.read_text()))
(prov/'excluded_attempts.json').write_text(json.dumps(failures,indent=2)+'\n')
code=prov/'source';code.mkdir(exist_ok=True)
workspace=Path(__file__).resolve().parents[1]
files=['dataset_generation/peg_tasks.py','scripts/collect_peg_quality.py','scripts/replay_peg_quality.py',
       'scripts/build_peg_mixture.py','scripts/export_peg_lerobot.py','scripts/validate_peg_lerobot.py','scripts/package_peg_dataset.py','scripts/runtime.py']
for name in files:
    dest=code/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(workspace/name,dest)
for pth in (pool/'source').glob('*.py'):shutil.copy2(pth,code/pth.name)
for name in ['peg_simulation.freeze.txt','peg_export.freeze.txt']:
    shutil.copy2(workspace/'requirements'/name,prov/name)
installed = workspace/'.envs/maniskill/lib/python3.10/site-packages/mani_skill'
for relative in ['envs/tasks/tabletop/lift_peg_upright.py','envs/tasks/tabletop/peg_insertion_side.py',
                 'examples/motionplanning/panda/motionplanner.py',
                 'examples/motionplanning/two_finger_gripper/motionplanner.py',
                 'examples/motionplanning/base_motionplanner/motionplanner.py']:
    dest=code/'installed_mani_skill'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(installed/relative,dest)
source_hashes={m['source_code_sha256'] for m in raw_manifests}
for digest in source_hashes:
    path=code/f'peg_tasks_{digest}.py';assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest
versions={name:importlib.metadata.version(name) for name in ['lerobot','numpy','torch','h5py','datasets','pyarrow','av']}
versions['simulation']=json.loads((workspace/'outputs/preflight/peg_dataset_feasibility/versions.json').read_text())
(prov/'versions.json').write_text(json.dumps(versions,indent=2)+'\n')
previews=root/'previews';previews.mkdir(exist_ok=True)
preview_sources={}
for condition in ['clean','stop','recovery','detour']:
    ep=next(e for e in manifest['episodes'] if e['condition']==condition)
    video=previews/f'{condition}.mp4'
    with h5py.File(replayed/condition/f"seed_{ep['seed']:04d}/trajectory.h5") as f, av.open(str(video),'w') as container:
        stream=container.add_stream('libx264',rate=20)
        stream.width=256;stream.height=128;stream.pix_fmt='yuv420p'
        stream.options={'crf':'23','preset':'veryfast'}
        stream.codec_context.thread_count=1
        for i in range(ep['frames']):
            image=np.concatenate([f[f'traj_0/obs/sensor_data/{cam}/rgb'][i] for cam in ['base_camera','hand_camera']],axis=1)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image,format='rgb24')):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    preview_sources[condition]={'episode_index':ep['episode_index'],'seed':ep['seed'],'frames':ep['frames']}
(previews/'sources.json').write_text(json.dumps(preview_sources,indent=2)+'\n')
hashes={}
for pth in sorted(root.rglob('*')):
    if pth.is_file() and pth.name!='sha256.json':
        digest=hashlib.sha256()
        with pth.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
        hashes[str(pth.relative_to(root))]=digest.hexdigest()
(prov/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
task=json.loads((prov/'task.json').read_text())
(root/'README.md').write_text(f'''# {raw_manifests[0]['task']} mixed30 / 40 episodes

Native LeRobot v3 dataset. 28 clean, 4 stop, 4 recovery, 4 detour episodes.
Mixed percentage is by episode count, not frame count.

- Robot: panda_wristcam; controller: pd_joint_pos; action: 8D; qpos state: 9D.
- FPS: 20; frames: {validation['frames']}; cameras: base_camera and hand_camera, 128x128 RGB.
- Task instruction: {task['instruction']}
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
''')
# Include README after it has been written.
hashes['README.md']=hashlib.sha256((root/'README.md').read_bytes()).hexdigest()
(prov/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps({'root':str(root),'validated_episodes':40,'frames':validation['frames'],
                  'excluded_attempts':len(failures),'generator_snapshots':sorted(source_hashes)},indent=2))
