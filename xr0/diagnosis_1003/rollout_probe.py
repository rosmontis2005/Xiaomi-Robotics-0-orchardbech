#!/usr/bin/env python3
"""Isolated diagnostic harness: unchanged production policy/controller, paired schedules."""
import bootstrap
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.spatial.transform import Rotation
from treesim.vla_env import OrchardVLAEnv, VLAEnvConfig
from observers import snapshot_grasp, snapshot_command, joint_command_delta

ROOT, XR0, ORCHARD = bootstrap.ROOT, bootstrap.XR0, bootstrap.ORCHARD
COHORT = [(2010027, 'episode_000010'), (2010042, 'episode_000020'), (2010067, 'episode_000030')]

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

bench = module(XR0/'tools/eval_orchard_learning_curve.py', 'diagnostic_benchmark_helpers')
frozen = module(ORCHARD/'log/v1_2650_collection_replay_filter/sources/frozen_gate1_oracle.py', 'diagnostic_frozen_oracle')
CONFIG = VLAEnvConfig(max_control_steps=900, detach_force_scale=1.5, grasp_mode='benchmark_assist')
frozen.CONFIG = CONFIG

def serial(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)

def save(path, data):
    with path.open('w') as f:
        json.dump(data, f, default=serial, allow_nan=False, indent=2)
        f.write('\n')

def line(stream, data):
    stream.write(json.dumps(data, default=serial, allow_nan=False)+'\n')

def obs_identity(obs):
    return {k: (hashlib.sha256(v.tobytes()).hexdigest() if k.startswith('rgb_') else v)
            for k, v in obs.items()}

def episode(policy, provider, mode, seed, episode_id, video):
    directory = ROOT/'rollouts'/f'{provider}_{mode}_{seed}'
    directory.mkdir(parents=True, exist_ok=False)
    annotation = ORCHARD/f'data/orchard_v1_2650/filtered/json/val/{episode_id}.json'
    traj = json.loads(annotation.read_text())
    gt_chunks = list(frozen.chunks(traj)) if provider == 'gt' else None
    row = bench.new_episode(dict(checkpoint=provider, checkpoint_path=str(annotation) if provider=='gt' else str(XR0/'outputs/orchard_v1_2650/project_orchardbench/orchard_v1_2650_frozen_vlm_seed42/epoch=0-step=10000.ckpt')), seed)
    row.update(mode=mode, episode_id=episode_id, annotation=str(annotation),
               environment=asdict(CONFIG), chunks_loaded=0, reached_advances=0,
               dwell_timeouts=0, control_budget=900, source='diagnosis_1003/rollout_probe.py')
    env = OrchardVLAEnv(CONFIG)
    recorder = bench.EpisodeVideo(video)
    started = time.monotonic()
    stage = 'reset'
    errors_pos, errors_rot, dwell_values = [], [], []
    with (directory/'steps.jsonl').open('x') as steps, (directory/'chunks.jsonl').open('x') as chunks:
        try:
            obs, info = env.reset(seed=seed)
            bench.observe_info(row, info)
            save(directory/'initial.json', dict(obs=obs_identity(obs), info=info, grasp=snapshot_grasp(env)))
            recorder.capture(obs)
            chunk_id = -1
            done = False
            while not done:
                chunk_id += 1
                stage = 'provide_chunk'
                anchor_obs = obs_identity(obs)
                prediction_seconds = 0.
                if provider == 'gt':
                    if chunk_id >= len(gt_chunks):
                        row['termination'] = 'target_sequence_exhausted'
                        break
                    first, end, anchor, adapter = gt_chunks[chunk_id]
                    k, stop = first-anchor, end-anchor
                    anchor_obs = dict(recorded_frame=anchor, tcp_pos_world=traj['proprios']['ee_pos'][anchor],
                                      tcp_rotm_world=traj['proprios']['ee_rotm'][anchor])
                else:
                    tic = time.monotonic()
                    policy.predict(obs, seed=42)
                    prediction_seconds = time.monotonic()-tic
                    row['replans'] += 1
                    adapter = policy.adapter
                    k, stop, anchor = 0, 30, None
                row['chunks_loaded'] += 1
                p, r, w = adapter.targets
                line(chunks, dict(chunk_id=chunk_id, at_control_step=row['control_steps'],
                                  anchor=anchor_obs, start_k=k, stop_k=stop,
                                  target_positions=p, target_rotations=r, target_widths=w,
                                  prediction_seconds=prediction_seconds))
                chunks.flush()
                chunk_steps, dwell = 0, 0
                while k < stop:
                    stage = 'observe_before'
                    pre_grasp = snapshot_grasp(env)
                    pre_q = env._joint_command.copy()
                    pre_pose = dict(position=obs['tcp_pos_world'], quaternion=obs['tcp_quat_world'],
                                    gripper_width=obs['gripper_width'], joint_pos=obs['joint_pos'])
                    command = adapter.to_native(k, obs)
                    command_diag = snapshot_command(env, command)
                    stage = 'environment_step'
                    obs, _, terminated, truncated, info = env.step(**command)
                    row['control_steps'] += 1
                    chunk_steps += 1
                    dwell += 1
                    bench.observe_info(row, info, stepped=True)
                    stage = 'observe_after'
                    post_grasp = snapshot_grasp(env)
                    pe = float(np.linalg.norm(p[k]-obs['tcp_pos_world']))
                    re = float((Rotation.from_matrix(r[k])*Rotation.from_quat(obs['tcp_quat_world']).inv()).magnitude())
                    reached = pe <= .01 and re <= .08
                    expired = mode == 'reach-conditioned' and not reached and dwell >= 30
                    advance = mode == 'time-indexed' or reached or expired
                    reason = 'time' if mode == 'time-indexed' else 'reached' if reached else 'maximum_dwell' if expired else 'hold'
                    row['reached_advances'] += int(advance and reached)
                    row['dwell_timeouts'] += int(expired)
                    errors_pos.append(pe); errors_rot.append(re)
                    target_index = anchor+k if provider == 'gt' else None
                    record = dict(control_step=row['control_steps'], sim_time=obs['sim_time'],
                        chunk_id=chunk_id, chunk_control_steps=chunk_steps, target_k=k,
                        gt_target_index=target_index, gt_phase=frozen.phase_at(traj,target_index) if provider=='gt' else None,
                        dwell_steps=dwell, reached=reached, advance=advance, advance_reason=reason,
                        target_position=p[k], target_rotation=r[k], target_width=w[k],
                        before_pose=pre_pose, measured_position=obs['tcp_pos_world'],
                        measured_quaternion=obs['tcp_quat_world'], measured_width=obs['gripper_width'],
                        position_error_m=pe, rotation_error_rad=re,
                        requested_command=command, command_diagnostics=command_diag,
                        grasp_before=pre_grasp, grasp_after=post_grasp,
                        observer_consistency=dict(clipping=command_diag['any_action_clipping']==bool(info['action_clipped']),
                            intent=command_diag['predicted_gripper_intent']==post_grasp['gripper_intent'],
                            open_steps=command_diag['predicted_width_open_steps']==post_grasp['width_open_steps']),
                        joint_command_delta=joint_command_delta(pre_q,env._joint_command,
                            max_joint_step=CONFIG.max_joint_step,action_repeat=CONFIG.action_repeat),
                        info=info)
                    line(steps,record)
                    recorder.capture(obs)
                    if row['control_steps']%150 == 0:
                        steps.flush()
                        print(json.dumps(dict(provider=provider,mode=mode,seed=seed,steps=row['control_steps'],
                             chunk=chunk_id,k=k,grasp=row['ever_grasped'],success=row['success'])),flush=True)
                    done = bool(terminated or truncated)
                    if done:
                        row['termination'] = 'success' if info['success'] else 'episode_budget'
                        break
                    if advance:
                        dwell_values.append(dwell)
                        k += 1
                        dwell = 0
                row.setdefault('chunk_control_steps', []).append(chunk_steps)
            row['capability_stage'] = bench.capability_stage(row)
        except Exception as exc:
            bench.mark_error(row,exc,stage)
            row['traceback'] = traceback.format_exc()
            print(row['traceback'],flush=True)
        finally:
            steps.flush(); chunks.flush()
            env.close()
    row['wall_seconds'] = time.monotonic()-started
    row['position_error_m'] = frozen.stats(errors_pos)
    row['rotation_error_rad'] = frozen.stats(errors_rot)
    row['completed_target_dwell'] = frozen.stats(dwell_values)
    for name in ('ik_failed','action_clipped'):
        row[name+'_rate'] = row[name+'_steps']/row['control_steps'] if row['control_steps'] else None
    recorder.finish(row,directory/'videos')
    save(directory/'summary.json',row)
    print('EPISODE_RESULT '+json.dumps(row,default=serial),flush=True)
    return row

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--provider',choices=['gt','step_10000'],required=True)
    ap.add_argument('--modes',nargs='+',choices=['time-indexed','reach-conditioned'],default=['reach-conditioned','time-indexed'])
    ap.add_argument('--seeds',type=int,nargs='+',default=[x[0] for x in COHORT])
    ap.add_argument('--no-video',action='store_true')
    args=ap.parse_args()
    policy=None
    if args.provider=='step_10000':
        from mibot.server.orchard_policy import OrchardPolicy
        weight=XR0/'outputs/orchard_v1_2650/project_orchardbench/orchard_v1_2650_frozen_vlm_seed42/epoch=0-step=10000.ckpt'
        policy=OrchardPolicy(str(weight),str(XR0.parent/'checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D'),
                            str(ORCHARD/'data/orchard_v1_2650/filtered/action_stats.json'))
        save(ROOT/'model_load_report.json',policy.load_report)
    results=[]
    for seed,ep in COHORT:
        if seed not in args.seeds: continue
        for mode in args.modes:
            results.append(episode(policy,args.provider,mode,seed,ep,not args.no_video))
    if any(r['termination']=='error' for r in results): raise SystemExit(1)

if __name__=='__main__': main()
