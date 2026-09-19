"""Seed-paired four-behavior Panda demonstrations with measured quality gates."""
import json
import hashlib
from pathlib import Path
import numpy as np
import gymnasium as gym
import sapien
import torch
import mani_skill.envs
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb

TASKS = {'lift': 'LiftPegUpright-v1', 'insertion': 'PegInsertionSide-v1'}
CONDITIONS = ('clean', 'stop', 'recovery', 'detour')
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
KWARGS = dict(robot_uids='panda_wristcam', obs_mode='none', control_mode='pd_joint_pos',
              reward_mode='sparse', render_mode=None, sim_backend='physx_cpu',
              render_backend='none', num_envs=1, max_episode_steps=900)


def arr(x):
    return x.detach().cpu().numpy().copy() if hasattr(x, 'detach') else np.asarray(x).copy()


def pose(x):
    raw = arr(x.raw_pose).reshape(-1, 7)[0]
    return sapien.Pose(raw[:3], raw[3:])


def tree_numpy(x):
    return {k: tree_numpy(v) for k, v in x.items()} if isinstance(x, dict) else arr(x)[0]


class QualityError(RuntimeError):
    pass


class TraceEnv(gym.Wrapper):
    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self.actions, self.states, self.trace = [], [], []
        self.events = {}
        self.sample()
        b = self.unwrapped
        self.initial = {'qpos': self.trace[0]['qpos'].tolist(), 'peg': self.trace[0]['peg'].tolist()}
        if hasattr(b, 'box'):
            self.initial.update(box=arr(b.box.pose.raw_pose)[0].tolist(),
                                peg_half_sizes=arr(b.peg_half_sizes)[0].tolist(),
                                hole_radius=arr(b.box_hole_radii).tolist(),
                                hole_offset=arr(b.box_hole_offsets.raw_pose)[0].tolist())
        return result

    def sample(self, result=None):
        b = self.unwrapped
        q = arr(b.peg.pose.q)[0]
        # A physical axis diagnostic independent of the task's Euler convention.
        mat = pose(b.peg.pose).to_transformation_matrix()[:3, :3]
        d = dict(tcp=arr(b.agent.tcp.pose.raw_pose)[0], peg=arr(b.peg.pose.raw_pose)[0],
                 qpos=arr(b.agent.robot.get_qpos())[0],
                 grasp=bool(b.agent.is_grasping(b.peg).item()),
                 success=bool(b.evaluate()['success'].item()),
                 peg_axis_vertical=abs(float(mat[2, 0])),
                 linear_velocity=arr(b.peg.linear_velocity)[0],
                 angular_velocity=arr(b.peg.angular_velocity)[0],
                 terminated=False if result is None else bool(result[2].item()),
                 truncated=False if result is None else bool(result[3].item()))
        if hasattr(b, 'box'):
            d.update(head_in_hole=arr(b.evaluate()['peg_head_pos_at_hole'])[0],
                     box=arr(b.box.pose.raw_pose)[0],
                     contact_force=float(np.linalg.norm(arr(b.scene.get_pairwise_contact_forces(b.peg, b.box)))))
        self.trace.append(d)
        self.states.append(tree_numpy(b.get_state_dict()))

    def step(self, action):
        result = super().step(action)
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.sample(result)
        if bool(result[3].item()): raise QualityError('horizon exhausted')
        return result

    def event(self, name, fn):
        start = len(self.actions)
        fn()
        self.events[name] = [start, len(self.actions)]

    def arrays(self):
        return {k: np.asarray([r[k] for r in self.trace]) for k in self.trace[0]}


def make_env(task):
    return TraceEnv(gym.make(TASKS[task], **KWARGS))


def solve(env, task, condition, seed):
    env.reset(seed=seed)
    b = env.unwrapped
    planner = PandaArmMotionPlanningSolver(env, vis=False, debug=False, base_pose=b.agent.robot.pose,
        visualize_target_grasp_pose=False, print_env_info=False, joint_vel_limits=0.75, joint_acc_limits=0.75)
    commanded_pose = pose(b.agent.tcp.pose)
    waypoint_targets = {}

    def move(target, refine=4):
        nonlocal commanded_pose
        r = planner.move_to_pose_with_screw(target, refine_steps=refine)
        if isinstance(r, int) and r == -1: raise QualityError('planning failed')
        commanded_pose = target

    def hold(frames):
        action = np.r_[arr(b.agent.robot.get_qpos())[0, :7], planner.gripper_state]
        for _ in range(frames): env.step(action)

    def stop(name):
        if condition == 'stop': env.event(name, lambda: hold(20))

    approaching = np.array([0, 0, -1])
    target_closing = pose(b.agent.tcp.pose).to_transformation_matrix()[:3, 1]
    grasp = compute_grasp_info_by_obb(get_actor_obb(b.peg), approaching=approaching,
                                    target_closing=target_closing, depth=0.025)
    grasp_pose = b.agent.build_grasp_pose(approaching, grasp['closing'], grasp['center'])
    offset = 0.10 if task == 'lift' else -max(0.05, float(b.peg_half_sizes[0, 0])/2+0.01)
    grasp_pose = grasp_pose * sapien.Pose([offset, 0, 0])

    if task == 'lift' and condition == 'recovery':
        # Miss the narrow side of the peg, not its long end; record an empty lift.
        miss = sapien.Pose([0, 0.09, 0]) * grasp_pose
        def fail_grasp():
            move(miss * sapien.Pose([0, 0, -0.05]))
            move(miss)
            planner.close_gripper(gripper_state=-0.6)
            move(sapien.Pose([0, 0, 0.10]) * miss)
        env.event('failed_attempt', fail_grasp)
        if b.agent.is_grasping(b.peg).item(): raise QualityError('miss grasp unexpectedly succeeded')
        if np.linalg.norm(env.trace[-1]['peg'][:3]-env.trace[0]['peg'][:3]) > 0.01:
            raise QualityError('miss displaced peg')
        env.event('retreat', lambda: move(sapien.Pose([0, 0, 0.18]) * miss))
        planner.open_gripper()
        env.events['retry'] = [len(env.actions), len(env.actions)]

    stop('stop_before_grasp')
    move(grasp_pose * sapien.Pose([0, 0, -0.05]))
    move(grasp_pose)
    planner.close_gripper(gripper_state=-0.6 if task == 'lift' else -1)
    lift_pose = sapien.Pose([0, 0, 0.30 if task == 'lift' else 0.13]) * grasp_pose
    move(lift_pose)
    if not b.agent.is_grasping(b.peg).item(): raise QualityError('grasp lost after lift')
    hold(10)  # Let the grasp settle before the measured stop interval.
    stop('stop_after_lift')

    if condition == 'detour':
        start_pose = pose(b.agent.tcp.pose)
        offsets = [[0.12, -0.16, 0.04], [-0.12, -0.16, 0.04]] if task == 'lift' else [[0.12, 0, 0.12], [-0.12, 0, 0.12]]
        targets = [sapien.Pose(offset)*start_pose for offset in offsets]
        for i, target in enumerate(targets):
            waypoint_targets[f'waypoint_{i}'] = target.p.tolist()
            env.event(f'waypoint_{i}', lambda target=target: move(target))
            if np.linalg.norm(arr(b.agent.tcp.pose.p)[0]-target.p) > 0.015:
                raise QualityError('waypoint tracking error')
            if not b.agent.is_grasping(b.peg).item(): raise QualityError('detour lost grasp')
        move(start_pose)

    if task == 'lift':
        theta = np.pi/10
        final_pose = lift_pose * sapien.Pose(q=[np.cos(theta), 0, np.sin(theta), 0])
        move(final_pose)
        move(sapien.Pose([0, 0, -0.10]) * final_pose)
        planner.open_gripper()
        env.event('settle', lambda: hold(20))
    else:
        half = float(b.peg_half_sizes[0, 0])
        def align(head_x=None, lateral=0., vertical=0.):
            if head_x is None: head_x = -half-0.035
            desired_peg = pose(b.box_hole_pose) * sapien.Pose([head_x-half, lateral, vertical])
            # Accumulate residual compensation against the commanded TCP target.
            # Using the measured TCP each time leaves a repeatable ~2.5mm sag,
            # nearly the whole 3mm insertion clearance.
            target = desired_peg * pose(b.peg.pose).inv() * commanded_pose
            move(target, refine=12)
        # Corrections while outside the hole, based on measured peg-in-gripper pose.
        align(vertical=0.15)
        for _ in range(6): align()
        if condition == 'recovery':
            align(lateral=0.012)
            env.event('failed_attempt', lambda: align(head_x=0.025, lateral=0.012))
            a, z = env.events['failed_attempt']
            if b.evaluate()['success'].item(): raise QualityError('failed insertion unexpectedly succeeded')
            if max(r['contact_force'] for r in env.trace[a:z+1]) < 0.2:
                raise QualityError('failed insertion did not contact box')
            commanded_pose = pose(b.agent.tcp.pose)  # Clear blocked insertion setpoint before retreat.
            env.event('retreat', lambda: align(head_x=-half-0.08, lateral=0.012))
            env.events['retry'] = [len(env.actions), len(env.actions)]
            for _ in range(6): align()
        def insert():
            for head_x in np.linspace(-half-0.025, 0.025, 10):
                align(head_x=float(head_x))
        env.event('insert', insert)
        env.event('settle', lambda: hold(20))

    trace = env.arrays()
    if not trace['success'][-10:].all(): raise QualityError('final success not stable for 10 frames')
    if task == 'lift' and (trace['peg_axis_vertical'][-10:] < 0.99).any():
        raise QualityError('peg not physically upright')
    if 'retry' in env.events: env.events['retry'][1] = len(env.actions)
    env.events['final_success'] = [len(env.actions)-10, len(env.actions)]
    if condition == 'stop':
        for key in ('stop_before_grasp', 'stop_after_lift'):
            start, end = env.events[key]
            for obj in ('tcp', 'peg'):
                drift = np.linalg.norm(trace[obj][start:end+1, :3]-trace[obj][start, :3], axis=1).max()
                if drift > 0.005: raise QualityError(f'{key} {obj} drift {drift}')
    if condition == 'detour':
        start = env.events['waypoint_0'][0]
        end = env.events['waypoint_1'][1]
        if not trace['grasp'][start:end+1].all(): raise QualityError('detour grasp not maintained')
    path = float(np.linalg.norm(np.diff(trace['tcp'][:, :3], axis=0), axis=1).sum())
    return dict(task=TASKS[task], condition=condition, seed=seed, steps=len(env.actions),
                initial=env.initial, events=env.events, tcp_path_m=path,
                success=True, waypoint_targets=waypoint_targets, source_code_sha256=SOURCE_SHA256,
                final_eval={k: arr(v).tolist() for k, v in b.evaluate().items()})


def save_episode(env, report, directory):
    import h5py
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    arrays = env.arrays()
    np.savez_compressed(directory/'trace.npz', **arrays)
    def stack_tree(items):
        if isinstance(items[0], dict): return {k: stack_tree([x[k] for x in items]) for k in items[0]}
        return np.stack(items)
    def write_tree(group, tree):
        for key, value in tree.items():
            if isinstance(value, dict): write_tree(group.create_group(key), value)
            else: group.create_dataset(key, data=value, compression='gzip')
    with h5py.File(directory/'trajectory.h5', 'w') as f:
        g = f.create_group('traj_0')
        write_tree(g, dict(actions=np.asarray(env.actions), terminated=arrays['terminated'][1:],
                          truncated=arrays['truncated'][1:], success=arrays['success'][1:],
                          env_states=stack_tree(env.states), obs={'agent': {'qpos': arrays['qpos']}}))
    metadata = {'env_info': {'env_id': report['task'], 'max_episode_steps': 900, 'env_kwargs': KWARGS},
                'source_type': 'motionplanning', 'source_desc': 'four-behavior peg dataset with trace quality gates',
                'episodes': [{'episode_id': 0, 'episode_seed': report['seed'], 'reset_kwargs': {'seed': report['seed']},
                              'control_mode': 'pd_joint_pos', 'elapsed_steps': report['steps'], 'info': {'success': True}}]}
    (directory/'trajectory.json').write_text(json.dumps(metadata, indent=2))
    (directory/'manifest.json').write_text(json.dumps(report, indent=2))
