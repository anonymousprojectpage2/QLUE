"""Audit native LeRobot files, every video, and exact source state/action rows."""
import argparse
from collections import Counter
import json
from pathlib import Path
from runtime import configure
configure()
import av
import h5py
import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset

p=argparse.ArgumentParser()
p.add_argument('--root',required=True)
p.add_argument('--repo-id',required=True)
p.add_argument('--replayed',required=True)
p.add_argument('--expected-episodes',type=int,default=40)
args=p.parse_args(); root=Path(args.root).resolve()
manifest=json.loads((root/'provenance/mixture_manifest.json').read_text())
task=json.loads((root/'provenance/task.json').read_text())
info=json.loads((root/'meta/info.json').read_text())
assert info['codebase_version']=='v3.0' and info['fps']==20
assert info['total_episodes']==args.expected_episodes==len(manifest['episodes'])
assert info['features']['action']['shape']==[8]
assert info['features']['observation.state']['shape']==[9]
counts=Counter(e['condition'] for e in manifest['episodes'])
if args.expected_episodes==40:
    assert counts=={'clean':28,'stop':4,'recovery':4,'detour':4}
    assert len({e['seed'] for e in manifest['episodes']})==40
ds=LeRobotDataset(args.repo_id,root=root,video_backend='pyav',delta_timestamps={'action':[i/20 for i in range(50)]})
rows=ds.hf_dataset.with_format('numpy')
offset=0; episodes=[]
for ep in manifest['episodes']:
    n=ep['frames']; block=rows[offset:offset+n]
    assert np.array_equal(np.asarray(block['index']).reshape(-1),np.arange(offset,offset+n))
    assert np.all(np.asarray(block['episode_index'])==ep['episode_index'])
    assert np.array_equal(np.asarray(block['frame_index']).reshape(-1),np.arange(n))
    assert np.allclose(np.asarray(block['timestamp']).reshape(-1),np.arange(n)/20,atol=1e-5)
    replay=Path(args.replayed)/ep['condition']/f"seed_{ep['seed']:04d}"
    with h5py.File(replay/'trajectory.h5') as f:
        assert np.array_equal(block['action'],f['traj_0/actions'][:])
        assert np.array_equal(block['observation.state'],f['traj_0/obs/agent/qpos'][:-1])
    for idx in [offset,offset+n//2,offset+n-1]:
        sample=ds[idx]
        assert sample['task']==task['instruction']
        assert tuple(sample['action'].shape)==(50,8)
        for cam in ['base_camera','hand_camera']:
            image=sample[f'observation.images.{cam}']
            assert tuple(image.shape)==(3,128,128)
            assert bool(image.isfinite().all()) and image.min()>=0 and image.max()<=1
    episodes.append({'episode':ep['episode_index'],'frames':n,'exact_state_action_match':True,'decoded_samples':3})
    offset+=n
assert offset==info['total_frames']==len(ds)
stats=json.loads((root/'meta/stats.json').read_text())
numeric_stats={}
numeric=ds.hf_dataset.select_columns(['action','observation.state']).with_format('numpy')[:]
for key in ['action','observation.state']:
    values=np.asarray(numeric[key],dtype=np.float64)
    assert np.isfinite(values).all()
    assert stats[key]['count']==[offset]
    errors={}
    for metric in ['mean','std','min','max']:
        expected=getattr(np,metric)(values,axis=0)
        actual=np.asarray(stats[key][metric])
        assert np.isfinite(actual).all() and np.allclose(actual,expected,atol=1e-5,rtol=1e-4),(key,metric)
        errors[metric+'_max_abs_error']=float(np.max(np.abs(actual-expected)))
    numeric_stats[key]=errors
videos={}
for cam in ['base_camera','hand_camera']:
    total=0
    for path in sorted((root/'videos'/f'observation.images.{cam}').glob('**/*.mp4')):
        with av.open(str(path)) as container:
            stream=container.streams.video[0]; assert float(stream.average_rate)==20
            previous=None; count=0
            for frame in container.decode(video=0):
                assert (frame.width,frame.height)==(128,128)
                assert previous is None or frame.pts>previous
                previous=frame.pts;count+=1
            total+=count
    assert total==offset,(cam,total,offset)
    videos[cam]={'decoded_frames':total,'fps':20,'resolution':[128,128]}
report={'valid':True,'episodes':args.expected_episodes,'frames':offset,'source_counts':dict(counts),
        'state_dim':9,'action_dim':8,'action_chunk_shape':[50,8],'videos':videos,
        'numeric_statistics_verified':numeric_stats,'episode_checks':episodes}
(root/'provenance/validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='episode_checks'},indent=2))
