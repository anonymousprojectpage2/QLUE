"""Collect and validate a paired seed pool for the peg tasks."""
import argparse
import json
import os
from pathlib import Path
import sys
from runtime import configure, ROOT
configure()
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # SAPIEN still constructs visual materials with render_backend=none.
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from dataset_generation.peg_tasks import make_env, solve, save_episode, CONDITIONS, QualityError, SOURCE_SHA256
torch.set_num_threads(4)
p = argparse.ArgumentParser()
p.add_argument('--task', choices=['lift', 'insertion'], required=True)
p.add_argument('--seeds', default='0:5')
p.add_argument('--conditions', nargs='+', default=list(CONDITIONS), choices=CONDITIONS)
p.add_argument('--output', required=True)
p.add_argument('--target-clean', type=int, default=0)
p.add_argument('--target-paired', type=int, default=12)
args = p.parse_args()
root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=True)
(root/'source').mkdir(exist_ok=True)
source = (ROOT/'dataset_generation/peg_tasks.py').read_bytes()
import hashlib
assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
(root/'source'/f'peg_tasks_{SOURCE_SHA256}.py').write_bytes(source)
start, end = map(int, args.seeds.split(':'))
env = make_env(args.task)
try:
    for seed in range(start, end):
        sets = {c: {p.parent.name for p in (root/c).glob('*/manifest.json')} for c in CONDITIONS}
        paired_count = len(set.intersection(*sets.values()))
        if args.target_clean and len(sets['clean']) >= args.target_clean and paired_count >= args.target_paired:
            print('Target pool counts reached', flush=True)
            break
        clean = None
        for condition in args.conditions:
            if args.target_clean and paired_count >= args.target_paired and condition != 'clean': continue
            dest = root/condition/f'seed_{seed:04d}'
            if (dest/'manifest.json').exists():
                report = json.loads((dest/'manifest.json').read_text())
                if condition == 'clean': clean = report
                continue
            try:
                report = solve(env, args.task, condition, seed)
                if clean is not None:
                    if report['initial'] != clean['initial']: raise QualityError('initial pairing mismatch')
                    report['clean_path_ratio'] = report['tcp_path_m']/clean['tcp_path_m']
                    if condition == 'detour' and report['clean_path_ratio'] < 1.25:
                        raise QualityError('detour path ratio below 1.25')
                if condition == 'clean': clean = report
                save_episode(env, report, dest)
                print(json.dumps({k: report[k] for k in ['condition', 'seed', 'steps', 'tcp_path_m', 'success']}), flush=True)
            except (QualityError, RuntimeError, ValueError) as e:
                failure = root/'failed'/f'{condition}_{seed:04d}'
                failure.parent.mkdir(exist_ok=True)
                np.savez_compressed(str(failure)+'.npz', **env.arrays())
                Path(str(failure)+'.json').write_text(json.dumps({'seed': seed, 'condition': condition, 'error': str(e), 'events': env.events}, indent=2))
                print(json.dumps({'condition': condition, 'seed': seed, 'error': str(e)}), flush=True)
finally:
    env.close()
