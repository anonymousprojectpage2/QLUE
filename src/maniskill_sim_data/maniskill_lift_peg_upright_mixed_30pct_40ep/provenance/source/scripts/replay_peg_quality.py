"""Verify action replay, then render all recorded states into RGB HDF5."""
import argparse
import json
import os
from pathlib import Path
import sys
from runtime import configure, ROOT
configure()
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sys.path.insert(0, str(ROOT))
import gymnasium as gym
import h5py
import numpy as np
import torch
import imageio.v2 as imageio
from dataset_generation.peg_tasks import TASKS, KWARGS, arr
torch.set_num_threads(4)
p = argparse.ArgumentParser()
p.add_argument('--task', required=True, choices=list(TASKS))
p.add_argument('--input', required=True)
p.add_argument('--output', required=True)
p.add_argument('--seeds', help='comma-separated seed allow-list')
p.add_argument('--selection', help='mixture manifest with episodes[].source_dir')
p.add_argument('--video', action='store_true')
args = p.parse_args()
root = Path(args.input).resolve(); out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=True)
if args.selection:
    sources = [Path(x['source_dir'])/'manifest.json' for x in json.loads(Path(args.selection).read_text())['episodes']]
else:
    sources = sorted(root.glob('*/seed_*/manifest.json'))
if args.seeds:
    allowed = set(map(int, args.seeds.split(',')))
    sources = [p for p in sources if json.loads(p.read_text())['seed'] in allowed]
env = gym.make(TASKS[args.task], **{**KWARGS, 'obs_mode': 'rgb', 'render_backend': 'sapien_cuda:0', 'render_mode': 'sensors'})
reports = []

def load_states(group):
    return {k: load_states(v) if isinstance(v, h5py.Group) else v[:] for k, v in group.items()}

def index_tree(tree, i):
    return {k: index_tree(v, i) if isinstance(v, dict) else v[i] for k, v in tree.items()}

try:
    for source in sources:
        manifest = json.loads(source.read_text())
        dest = out/manifest['condition']/f"seed_{manifest['seed']:04d}"
        if (dest/'replay_validation.json').exists():
            reports.append(json.loads((dest/'replay_validation.json').read_text())); continue
        dest.mkdir(parents=True, exist_ok=True)
        with h5py.File(source.with_name('trajectory.h5'), 'r') as raw:
            g = raw['traj_0']; actions = g['actions'][:]; states = load_states(g['env_states'])
            raw_success = g['success'][:]
            trace = np.load(source.with_name('trace.npz'))
            env.reset(seed=manifest['seed'])
            b = env.unwrapped
            assert np.allclose(arr(b.peg.pose.raw_pose)[0], manifest['initial']['peg'], atol=1e-7)
            if args.task == 'insertion':
                assert np.array_equal(arr(b.peg_half_sizes)[0], manifest['initial']['peg_half_sizes'])
                assert np.array_equal(arr(b.box_hole_offsets.raw_pose)[0], manifest['initial']['hole_offset'])
            # Action-only replay: no intermediate state injection.
            b.set_state_dict(index_tree(states, 0))
            success, errors = [], []
            def diagnostics():
                d = {'tcp': arr(b.agent.tcp.pose.p)[0], 'peg': arr(b.peg.pose.p)[0],
                     'grasp': bool(b.agent.is_grasping(b.peg).item())}
                if hasattr(b, 'box'):
                    d['head'] = arr(b.evaluate()['peg_head_pos_at_hole'])[0]
                    d['contact'] = float(np.linalg.norm(arr(b.scene.get_pairwise_contact_forces(b.peg, b.box))))
                return d
            diagnostic = [diagnostics()]
            for i, action in enumerate(actions):
                _, _, _, trunc, info = env.step(action)
                assert not trunc.item()
                success.append(bool(info['success'].item()))
                errors.append(float(np.max(np.abs(arr(b.peg.pose.raw_pose)[0]-trace['peg'][i+1]))))
                diagnostic.append(diagnostics())
            if not all(success[-10:]): raise RuntimeError(f'action replay final success failed: {source}')
            events = manifest['events']
            if manifest['condition'] == 'stop':
                for key in ['stop_before_grasp', 'stop_after_lift']:
                    start, end = events[key]
                    for obj in ['tcp', 'peg']:
                        drift = max(np.linalg.norm(diagnostic[i][obj]-diagnostic[start][obj]) for i in range(start, end+1))
                        assert drift <= 0.005, (source, key, obj, drift)
            if manifest['condition'] == 'recovery':
                start, end = events['failed_attempt']; rs, re = events['retreat']
                assert not success[end-1]
                if 'head' in diagnostic[0]:
                    assert max(d['contact'] for d in diagnostic[start:end+1]) >= 0.2
                    assert diagnostic[rs]['head'][0]-diagnostic[re]['head'][0] >= 0.04
                else:
                    assert not diagnostic[end]['grasp']
                    assert np.linalg.norm(diagnostic[end]['peg']-diagnostic[0]['peg']) <= 0.01
            if manifest['condition'] == 'detour':
                start, end = events['waypoint_0'][0], events['waypoint_1'][1]
                assert all(d['grasp'] for d in diagnostic[start:end+1])
                path = np.linalg.norm(np.diff(np.stack([d['tcp'] for d in diagnostic]), axis=0), axis=1).sum()
                clean_manifest = json.loads((source.parent.parent.parent/'clean'/source.parent.name/'manifest.json').read_text())
                assert path/clean_manifest['tcp_path_m'] >= 1.25
            # State-based RGB extraction preserves the recorded condition semantics.
            env.reset(seed=manifest['seed'])
            with h5py.File(dest/'trajectory.h5', 'w') as rendered:
                raw.copy('traj_0', rendered)
                target = rendered['traj_0']
                sensors = target.require_group('obs/sensor_data')
                rgb = {k: sensors.require_group(k).create_dataset('rgb', shape=(len(actions)+1,128,128,3), dtype='uint8', compression='gzip', compression_opts=1)
                       for k in ['base_camera', 'hand_camera']}
                writer = imageio.get_writer(dest/'video.mp4', fps=20) if args.video else None
                try:
                    restored_success = []
                    for i in range(len(actions)+1):
                        b.set_state_dict(index_tree(states, i))
                        obs = b.get_obs()
                        restored_success.append(bool(b.evaluate()['success'].item()))
                        images = []
                        for camera in rgb:
                            image = arr(obs['sensor_data'][camera]['rgb'])[0]
                            rgb[camera][i] = image
                            images.append(image)
                        if writer is not None and i < len(actions): writer.append_data(np.concatenate(images, axis=1))
                    assert np.array_equal(restored_success[1:], raw_success)
                finally:
                    if writer is not None: writer.close()
            metadata = json.loads(source.with_name('trajectory.json').read_text())
            metadata['env_info']['env_kwargs'].update(obs_mode='rgb', render_backend='sapien_cuda:0', render_mode='sensors')
            (dest/'trajectory.json').write_text(json.dumps(metadata, indent=2))
            report = {'task': TASKS[args.task], 'condition': manifest['condition'], 'seed': manifest['seed'],
                      'frames': len(actions), 'action_replay_final_success': True,
                      'action_replay_peg_max_abs_pose_error': max(errors),
                      'action_replay_success_flags_equal': bool(np.array_equal(success, raw_success)),
                      'state_replay_success_flags_equal': True, 'source_dir': str(source.parent)}
            report['action_replay_behavior_valid'] = True
            (dest/'replay_validation.json').write_text(json.dumps(report, indent=2))
            reports.append(report)
            print(json.dumps(report), flush=True)
        (out/f'validation_worker_{os.getpid()}.json').write_text(json.dumps(reports, indent=2))
finally:
    env.close()
