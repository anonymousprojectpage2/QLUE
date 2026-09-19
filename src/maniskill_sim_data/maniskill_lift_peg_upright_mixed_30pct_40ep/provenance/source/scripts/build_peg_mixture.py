"""Select 40 unique seeds with 28/4/4/4 sources and verify raw traces."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--pool', required=True)
p.add_argument('--output', required=True)
args = p.parse_args()
pool = Path(args.pool).resolve(); out = Path(args.output).resolve()
if out.exists(): raise FileExistsError(out)
conditions = ['clean', 'stop', 'recovery', 'detour']
sources = {c: {int(p.parent.name.split('_')[1]): p.parent for p in (pool/c).glob('seed_*/manifest.json')} for c in conditions}
clean_seeds = set(sources['clean'])
paired = sorted(set.intersection(*(set(sources[c]) for c in conditions)))
if len(clean_seeds) < 40 or len(paired) < 12:
    raise ValueError(f'Need clean>=40 and fully paired>=12; clean={len(clean_seeds)}, paired={len(paired)}')
rng = np.random.default_rng(20260913)
variant_seeds = rng.permutation(paired)[:12].tolist()
normal_seeds = rng.permutation(sorted(clean_seeds-set(variant_seeds)))[:28].tolist()
labels = rng.permutation(['stop']*4+['recovery']*4+['detour']*4).tolist()
assignment = {**dict(zip(variant_seeds, labels)), **{s: 'clean' for s in normal_seeds}}
episodes = []
for index, seed in enumerate(sorted(assignment)):
    condition = assignment[seed]; directory = sources[condition][seed]
    m = json.loads((directory/'manifest.json').read_text())
    clean = json.loads((sources['clean'][seed]/'manifest.json').read_text())
    assert m['initial'] == clean['initial']
    trace = np.load(directory/'trace.npz')
    with h5py.File(directory/'trajectory.h5') as f:
        g = f['traj_0']; actions = g['actions'][:]
        assert actions.shape == (m['steps'], 8) and np.isfinite(actions).all()
        assert trace['qpos'].shape == (m['steps']+1, 9)
        assert not trace['truncated'].any() and trace['success'][-10:].all()
        assert np.array_equal(g['success'][:], trace['success'][1:])
    events = m['events']; details = {}
    if condition == 'stop':
        for name in ['stop_before_grasp', 'stop_after_lift']:
            start, end = events[name]; assert end-start == 20
            for obj in ['tcp', 'peg']:
                drift = float(np.linalg.norm(trace[obj][start:end+1, :3]-trace[obj][start, :3], axis=1).max())
                assert drift <= 0.005
                details[name+'_'+obj+'_drift_m'] = drift
    if condition == 'detour':
        ratio = m['tcp_path_m']/clean['tcp_path_m']; assert ratio >= 1.25
        assert 'waypoint_0' in events and 'waypoint_1' in events
        for key in ['waypoint_0', 'waypoint_1']:
            start, end = events[key]; assert trace['grasp'][end]
            if key in m.get('waypoint_targets', {}):
                distance = float(np.linalg.norm(trace['tcp'][end,:3]-m['waypoint_targets'][key]))
                assert distance <= 0.015
                details[key+'_tracking_error_m'] = distance
        details['path_ratio'] = ratio
    if condition == 'recovery':
        start, end = events['failed_attempt']; rs, re = events['retreat']
        assert start < end <= rs < re <= events['retry'][0]
        assert not trace['success'][end]
        if 'head_in_hole' in trace:
            peak = float(trace['contact_force'][start:end+1].max()); assert peak >= 0.2
            retreat = float(trace['head_in_hole'][rs, 0]-trace['head_in_hole'][re, 0]); assert retreat >= 0.04
            details.update(failed_peak_contact_force=peak, retreat_m=retreat)
        else:
            assert not trace['grasp'][end]
            peg_motion = float(np.linalg.norm(trace['peg'][end,:3]-trace['peg'][0,:3]))
            assert peg_motion <= 0.01
            details['failed_grasp_peg_displacement_m'] = peg_motion
    episodes.append({'episode_index': index, 'seed': seed, 'condition': condition,
                     'source_dir': str(directory), 'clean_reference_dir': str(sources['clean'][seed]),
                     'frames': m['steps'], 'quality_checks': details,
                     'source_sha256': hashlib.sha256((directory/'trajectory.h5').read_bytes()).hexdigest()})
out.mkdir(parents=True)
report = {'sampling_seed': 20260913, 'source_counts': {'clean':28,'stop':4,'recovery':4,'detour':4},
          'pool_clean_count': len(clean_seeds), 'pool_fully_paired_count': len(paired),
          'selection_rule': 'Choose 12 fully paired seeds, then 28 distinct additional successful clean seeds; shuffle balanced variant labels.',
          'episodes': episodes}
(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
with h5py.File(out/'mixed30.h5','w') as f:
    import os
    for ep in episodes:
        f[f"traj_{ep['episode_index']}"] = h5py.ExternalLink(os.path.relpath(Path(ep['source_dir'])/'trajectory.h5',out),'traj_0')
metadata = json.loads((Path(episodes[0]['source_dir'])/'trajectory.json').read_text())
metadata['episodes'] = []
for ep in episodes:
    item = json.loads((Path(ep['source_dir'])/'trajectory.json').read_text())['episodes'][0]
    item['episode_id'] = ep['episode_index']; metadata['episodes'].append(item)
(out/'mixed30.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps({'output':str(out),'episodes':40,'counts':report['source_counts'],'paired_pool':len(paired)},indent=2))
