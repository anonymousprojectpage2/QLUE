"""Write native LeRobot v3 via the installed LeRobot writer, from replayed RGB."""
import argparse
import json
from pathlib import Path
import shutil
import time
from runtime import configure, ROOT
configure()
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

p=argparse.ArgumentParser()
p.add_argument('--selection',required=True)
p.add_argument('--replayed',required=True)
p.add_argument('--output',required=True)
p.add_argument('--repo-id',required=True)
p.add_argument('--task',required=True,choices=['lift','insertion'])
p.add_argument('--wait-for-replay',action='store_true',help='Wait up to 10 minutes for each replay validation marker')
args=p.parse_args()
selection=json.loads(Path(args.selection).read_text())
replayed=Path(args.replayed).resolve(); out=Path(args.output).resolve()
if out.exists(): raise FileExistsError(out)
features={'observation.state':{'dtype':'float32','shape':(9,),'names':[f'qpos_{i}' for i in range(9)]},
          'action':{'dtype':'float32','shape':(8,),'names':[f'joint_{i}' for i in range(7)]+['gripper']}}
for camera in ['base_camera','hand_camera']:
    features[f'observation.images.{camera}']={'dtype':'video','shape':(3,128,128),'names':['channels','height','width']}
ds=LeRobotDataset.create(repo_id=args.repo_id,root=out,fps=20,robot_type='panda_wristcam',features=features,
                        video_backend='pyav',vcodec='h264',streaming_encoding=True,encoder_threads=2)
instruction='Stand the peg upright on the table.' if args.task=='lift' else 'Insert the peg into the side hole.'
try:
    for ep in selection['episodes']:
        source=replayed/ep['condition']/f"seed_{ep['seed']:04d}"
        deadline=time.monotonic()+600
        while True:
            try:
                validation=json.loads((source/'replay_validation.json').read_text())
                break
            except (FileNotFoundError,json.JSONDecodeError):
                if not args.wait_for_replay or time.monotonic()>=deadline:raise
                time.sleep(1)
        assert validation['action_replay_final_success'] and validation['state_replay_success_flags_equal']
        if len(selection['episodes'])==40:
            assert validation.get('action_replay_behavior_valid') is True
        with h5py.File(source/'trajectory.h5') as f:
            g=f['traj_0']; n=len(g['actions']); assert n==ep['frames']
            for i in range(n):
                frame={'task':instruction,'observation.state':g['obs/agent/qpos'][i].astype('float32'),
                       'action':g['actions'][i].astype('float32')}
                for cam in ['base_camera','hand_camera']:
                    frame[f'observation.images.{cam}']=g[f'obs/sensor_data/{cam}/rgb'][i]
                ds.add_frame(frame)
            ds.save_episode(parallel_encoding=False)
        print(json.dumps({'episode':ep['episode_index'],'seed':ep['seed'],'condition':ep['condition'],'frames':n}),flush=True)
finally:
    ds.finalize()
(out/'provenance').mkdir()
shutil.copy2(args.selection,out/'provenance/mixture_manifest.json')
(out/'provenance/task.json').write_text(json.dumps({'task':args.task,'instruction':instruction,'control_mode':'pd_joint_pos',
                                                 'state':'Panda qpos: 7 arm joints + 2 fingers','fps':20},indent=2))
print('Export finished',out,flush=True)
