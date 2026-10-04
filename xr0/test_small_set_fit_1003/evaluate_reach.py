#!/usr/bin/env python3
"""M0 paired evaluation: original reach protocol, unchanged production behavior.

No GT target/identity/phase reaches a model input or a model control decision.
GT metadata is used only by the explicit gt provider and passive reset checks.
--validate performs CPU-only file/schema/protocol checks without simulator import.
"""
import argparse
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback
import bootstrap

ROOT, XR0, ORCHARD = bootstrap.ROOT, bootstrap.XR0, bootstrap.ORCHARD
BASE_CHECKPOINT = XR0/'outputs/orchard_v1_2650/project_orchardbench/orchard_v1_2650_frozen_vlm_seed42/epoch=0-step=10000.ckpt'
STATS = ORCHARD/'data/orchard_v1_2650/filtered/action_stats.json'
PROCESSOR = XR0.parent/'checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def advance_decision(position_error, rotation_error, dwell):
    reached = position_error <= .01 and rotation_error <= .08
    expired = not reached and dwell >= 30
    return reached, expired, reached or expired, 'reached' if reached else 'maximum_dwell' if expired else 'hold'


def selected_scenes(path):
    selection = json.loads(path.read_text())
    episodes = {row['episode_id']: row for row in selection['episodes']}
    selected = []
    for item in selection['closed_loop_scenes']:
        if isinstance(item, str):
            scene = dict(episodes[item])
        else:
            scene = dict(episodes[item['episode_id']]) | item
        selected.append(scene)
    if len({row['seed'] for row in selected}) != len(selected):
        raise ValueError('Closed-loop seeds must be unique')
    if sorted(row['split'] for row in selected) != ['train', 'train', 'val', 'val']:
        raise ValueError('Expected preselected 2 train + 2 val scenes')
    return selection, selected


def validate_inputs(selection_path):
    selection, scenes = selected_scenes(selection_path)
    checks = []
    for scene in scenes:
        path = Path(scene['annotation'])
        data = json.loads(path.read_text())
        if (int(data['seed']) != int(scene['seed']) or data['episode_id'] != scene['episode_id']
                or data['split'] != scene['split']):
            raise ValueError('Scene identity mismatch: ' + str(path))
        if sha256(path) != scene['annotation_sha256']:
            raise ValueError('Annotation changed after scene selection: ' + str(path))
        if data['num_frames'] < 30 or data['record_fps'] != 30 or data['orchardbench']['timestamps'][0] != 0.:
            raise ValueError('Unexpected trajectory timing/schema')
        if not all(data['orchardbench'][key] for key in ('first_attempt_success', 'grasped', 'detached', 'placed')):
            raise ValueError('Expected successful accepted teacher annotation')
        videos = []
        for key in ('ego', 'wrist_left'):
            video = Path(data['observations'][key][0]['path'])
            if not video.is_file():
                raise FileNotFoundError(video)
            videos.append(str(video))
        n = data['num_frames']
        consumed = []
        for first in range(0, n, 30):
            anchor = min(first, n - 30)
            consumed.extend(anchor+k for k in range(first-anchor, min(first+30, n)-anchor))
        if consumed != list(range(n)):
            raise AssertionError('GT tail-window coverage incorrect')
        checks.append(dict(seed=scene['seed'], episode_id=scene['episode_id'], split=scene['split'],
            annotation=str(path), annotation_sha256=sha256(path), num_frames=n, videos=videos,
            gt_target_indices_covered_exactly_once=True))
    assert advance_decision(.01, .08, 1) == (True, False, True, 'reached')
    assert advance_decision(.011, .08, 29) == (False, False, False, 'hold')
    assert advance_decision(.011, .08, 30) == (False, True, True, 'maximum_dwell')
    assert advance_decision(.01, .081, 30) == (False, True, True, 'maximum_dwell')
    stats = json.loads(STATS.read_text())
    if stats['contract'] != 'orchard_cartesian_local_rotvec_width_v1' or stats['source_split'] != 'train':
        raise ValueError('Wrong production action statistics')
    for path in (Path(__file__), ROOT/'bootstrap.py', XR0/'diagnosis_1003/observers.py'):
        ast.parse(path.read_text(), filename=str(path))
    if not BASE_CHECKPOINT.is_file():
        raise FileNotFoundError(BASE_CHECKPOINT)
    return dict(status='PASS', cpu_only=True, simulator_imported=False,
        selection_path=str(selection_path), selection_sha256=sha256(selection_path),
        action_stats_sha256=sha256(STATS), scenes=checks,
        reach_protocol=dict(position_tolerance_m=.01, rotation_tolerance_rad=.08, maximum_dwell=30,
                            control_budget=900, targets_per_chunk=30, no_new_pregrasp_gate=True),
        generated_utc_unix=time.time())


def load_runtime():
    global np, Rotation, OrchardVLAEnv, VLAEnvConfig, CONFIG, bench, frozen
    global snapshot_grasp, snapshot_command, joint_command_delta
    import numpy as np
    from scipy.spatial.transform import Rotation
    from treesim.vla_env import OrchardVLAEnv, VLAEnvConfig
    observers = module(XR0/'diagnosis_1003/observers.py', 'm0_observers')
    snapshot_grasp, snapshot_command, joint_command_delta = observers.snapshot_grasp, observers.snapshot_command, observers.joint_command_delta
    bench = module(XR0/'tools/eval_orchard_learning_curve.py', 'm0_benchmark_helpers')
    frozen = module(ORCHARD/'log/v1_2650_collection_replay_filter/sources/frozen_gate1_oracle.py', 'm0_frozen_oracle')
    CONFIG = VLAEnvConfig(max_control_steps=900, detach_force_scale=1.5, grasp_mode='benchmark_assist')
    frozen.CONFIG = CONFIG


def check_reset_scene(obs, info, traj, directory, initial):
    """Passive checks only; annotation metadata never changes policy execution."""
    from decord import VideoReader
    from PIL import Image
    mismatches, checks = [], {}
    p = traj['proprios']
    expected_rotation = np.asarray(p['ee_rotm'][0]).reshape(3,3)
    measured_rotation = Rotation.from_quat(obs['tcp_quat_world']).as_matrix()
    checks['reset_time_zero'] = obs['sim_time'] == 0.
    checks['position_matches_annotation'] = np.allclose(obs['tcp_pos_world'], p['ee_pos'][0], atol=2e-6, rtol=0)
    checks['rotation_matches_annotation'] = np.allclose(measured_rotation, expected_rotation, atol=2e-6, rtol=0)
    checks['arm_joints_match_annotation'] = np.allclose(obs['joint_pos'][:7], p['arm_joint'][0], atol=2e-6, rtol=0)
    checks['width_matches_annotation'] = abs(float(obs['gripper_width']) - float(p['gripper_pos'][0][0])) <= 2e-6
    stance = info['reset_stance_debug']
    expert = traj['orchardbench']['fixed_base_expert']
    checks['seed_matches_annotation'] = int(info['seed']) == int(traj['seed'])
    checks['planned_apple_identity_matches_annotation'] = int(stance['apple_index']) == int(expert['selected_apple_debug_index'])
    checks['planned_fruit_pose_matches_annotation'] = np.allclose(stance['target_world'], expert['selected_apple_initial_world_pose'][:3], atol=2e-6, rtol=0)
    checks['fixed_base_matches_annotation'] = np.allclose(list(stance['base_xy']) + [stance['base_yaw']], expert['selected_base_pose_xy_yaw'], atol=2e-6, rtol=0)
    peers = []
    for other in sorted((ROOT/'rollouts').glob(f'*_{traj["seed"]}/initial.json')):
        if other.parent == directory:
            continue
        previous = json.loads(other.read_text())
        peer_checks = {}
        for key, value in initial['obs'].items():
            peer_checks[key] = (value == previous['obs'][key] if key.startswith('rgb_')
                                else bool(np.array_equal(np.asarray(value), np.asarray(previous['obs'][key]))))
        peer_summary = json.loads((other.parent/'summary.json').read_text())
        peer_checks['scene_annotation_identity'] = (int(peer_summary['seed']) == int(traj['seed'])
            and peer_summary['episode_id'] == traj['episode_id']
            and peer_summary['split'] == traj['split']
            and peer_summary['annotation_sha256'] == sha256(Path(peer_summary['annotation'])))
        peer_checks['same_annotation_path'] = Path(peer_summary['annotation']).resolve() == Path(initial['annotation']).resolve()
        peers.append(dict(path=str(other), checks=peer_checks, exact_match=all(peer_checks.values()),
            full_selection_sha256_matches=previous.get('selection_sha256') == initial['selection_sha256'],
            previous_selection_sha256=previous.get('selection_sha256')))
    if any(not peer['exact_match'] for peer in peers):
        mismatches.append('same_seed_provider_reset_mismatch')
    mismatches += [name for name, okay in checks.items() if not okay]
    views = []
    montage = Image.new('RGB', (384, 288))
    for row, (view, obs_key) in enumerate((('ego', 'rgb_static'), ('wrist_left', 'rgb_wrist'))):
        path = Path(traj['observations'][view][0]['path'])
        reader = VideoReader(str(path), num_threads=1)
        if len(reader) != traj['num_frames']:
            mismatches.append(view + '_video_frame_count')
        decoded = reader[0].asnumpy()
        raw = obs[obs_key]
        if decoded.shape != raw.shape:
            mismatches.append(view + '_video_shape')
            continue
        difference = np.abs(decoded.astype(float) - raw.astype(float))
        views.append(dict(view=view, path=str(path), video_frames=len(reader),
            decoded_first_frame_sha256=hashlib.sha256(decoded.tobytes()).hexdigest(),
            reset_rgb_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
            raw_vs_compressed_mae_255=float(difference.mean()),
            raw_vs_compressed_abs_error_p95_255=float(np.percentile(difference,95))))
        montage.paste(Image.fromarray(raw), (0, row*144))
        montage.paste(Image.fromarray(decoded), (192, row*144))
    montage.save(directory/'reset_vs_source_video.png')
    return dict(state_and_identity_pass=not mismatches, checks=checks, mismatches=mismatches,
        paired_reset_checks=peers, source_video_checks=views,
        video_comparison_rule='Raw/reset versus compressed first video frame: report errors and montage; byte-identical hashes are not required.',
        montage_columns=['live_reset_rgb', 'decoded_teacher_video_frame_0'], montage_rows=['ego', 'wrist_left'])

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

def episode(policy, provider, scene, checkpoint_path, video, selection_sha256):
    mode = 'reach-conditioned'
    seed, episode_id = int(scene['seed']), scene['episode_id']
    directory = ROOT/'rollouts'/f'{provider}_{mode}_{seed}'
    directory.mkdir(parents=True, exist_ok=False)
    annotation = Path(scene['annotation'])
    traj = json.loads(annotation.read_text())
    gt_chunks = list(frozen.chunks(traj)) if provider == 'gt' else None
    if int(traj['seed']) != seed:
        raise ValueError('Selection/annotation seed mismatch')
    row = bench.new_episode(dict(checkpoint=provider, checkpoint_path=str(checkpoint_path)), seed)
    row.update(mode=mode, episode_id=episode_id, annotation=str(annotation),
               environment=asdict(CONFIG), chunks_loaded=0, reached_advances=0,
               dwell_timeouts=0, control_budget=900, source='test_small_set_fit_1003/evaluate_reach.py',
               split=scene['split'], selection_sha256=selection_sha256,
               annotation_sha256=sha256(annotation), inference_seed=42)
    env = None
    recorder = bench.EpisodeVideo(video)
    started = time.monotonic()
    stage = 'reset'
    errors_pos, errors_rot, dwell_values = [], [], []
    with (directory/'steps.jsonl').open('x') as steps, (directory/'chunks.jsonl').open('x') as chunks:
        try:
            env = OrchardVLAEnv(CONFIG)
            obs, info = env.reset(seed=seed)
            bench.observe_info(row, info)
            initial = dict(obs=obs_identity(obs), info=info, grasp=snapshot_grasp(env), selection_sha256=selection_sha256, annotation=str(annotation))
            save(directory/'initial.json', initial)
            scene_report = check_reset_scene(obs, info, traj, directory, initial)
            save(directory/'scene_check.json', scene_report)
            row['scene_check_pass'] = scene_report['state_and_identity_pass']
            if not scene_report['state_and_identity_pass']:
                raise ValueError('Reset scene/state mismatch: ' + json.dumps(scene_report['mismatches']))
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
                    normalized = policy.predict(obs, seed=42)
                    if tuple(normalized.shape) != (30, 32):
                        raise ValueError('Expected normalized action shape (30,32)')
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
                    reached, expired, advance, reason = advance_decision(pe, re, dwell)
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
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    bench.mark_error(row, exc, 'environment_close')
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, default=ROOT/'selection.json')
    parser.add_argument('--provider', choices=['gt', 'original_10k', 'best'])
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/best_trainable.pt')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--video', choices=['none', 'first', 'all'], default='first')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    validation = validate_inputs(args.selection)
    if args.validate:
        (ROOT/'evaluation_input_validation.json').write_text(json.dumps(validation, indent=2)+'\n')
        print(json.dumps(validation))
        return
    if args.provider is None:
        parser.error('--provider is required unless --validate is set')
    _, scenes = selected_scenes(args.selection)
    if args.seeds is not None:
        if not set(args.seeds) <= {int(s['seed']) for s in scenes}:
            raise ValueError('Requested seeds must belong to the preselected closed-loop scenes')
        scenes = [scene for scene in scenes if int(scene['seed']) in args.seeds]
    load_runtime()
    policy = None
    checkpoint_path = 'per-scene absolute world GT trajectory'
    load_report = None
    if args.provider != 'gt':
        from mibot.server.orchard_policy import OrchardPolicy
        policy = OrchardPolicy(str(BASE_CHECKPOINT), str(PROCESSOR), str(STATS))
        checkpoint_path = str(BASE_CHECKPOINT)
        load_report = dict(base=policy.load_report)
        if args.provider == 'best':
            from checkpoint_io import load_trainable_overlay
            load_report['overlay'] = load_trainable_overlay(policy.model, args.checkpoint, base_checkpoint=BASE_CHECKPOINT)
            checkpoint_path = str(args.checkpoint.resolve())
            policy.model.eval()
        from deployment_precision_probe import run_probe
        reference_step = 0 if args.provider == 'original_10k' else int(load_report['overlay']['step'])
        precision_report = run_probe(policy, provider=args.provider, reference_step=reference_step,
                                     checkpoint_path=checkpoint_path, load_report=load_report)
        print('DEPLOYMENT_PRECISION '+json.dumps(precision_report), flush=True)
    else:
        precision_report = None
    manifest = dict(provider=args.provider, checkpoint_path=checkpoint_path, load_report=load_report,
        deployment_precision_probe=precision_report,
        selection_sha256=validation['selection_sha256'], action_stats_sha256=validation['action_stats_sha256'],
        scenes=[dict(seed=s['seed'], episode_id=s['episode_id'], split=s['split']) for s in scenes],
        reach_protocol=validation['reach_protocol'], production_code_modified=False,
        source_sha256={str(p): sha256(p) for p in [Path(__file__), ROOT/'deployment_precision_probe.py',
            ROOT/'checkpoint_io.py', XR0/'diagnosis_1003/observers.py',
            ORCHARD/'treesim/orchard_action.py', ORCHARD/'treesim/vla_env.py', XR0/'mibot/server/orchard_policy.py']})
    manifest_path = ROOT/f'evaluation_manifest_{args.provider}.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['selection_sha256'] != manifest['selection_sha256'] or previous['checkpoint_path'] != manifest['checkpoint_path']:
            raise ValueError('Existing evaluation manifest refers to another selection/checkpoint')
    else:
        with manifest_path.open('x') as stream:
            json.dump(manifest, stream, default=serial, allow_nan=False, indent=2)
            stream.write('\n')
    results = []
    for index, scene in enumerate(scenes):
        video = args.video == 'all' or (args.video == 'first' and index == 0)
        results.append(episode(policy, args.provider, scene, checkpoint_path, video, validation['selection_sha256']))
    print('EVALUATION_COMPLETE '+json.dumps(dict(provider=args.provider, scenes=len(results),
        valid=sum(r['termination']!='error' for r in results),
        success=sum(r['success'] is True for r in results), grasp=sum(r['ever_grasped'] for r in results))))
    if any(r['termination']=='error' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
