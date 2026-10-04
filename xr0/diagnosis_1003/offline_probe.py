#!/usr/bin/env python3
"""Read-only expert-observation probe. All generated files stay beside this script.

--prepare performs CPU selection, RGB/contract checks, and mean-action baseline.
--infer loads the requested existing checkpoint(s); it never calls training code.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import bootstrap  # noqa: F401; diagnostic-local paths, offline mode and caches
import argparse
import ast
import copy
import gc
import hashlib
import json
import time
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from decord import VideoReader
from treesim.orchard_action import CONTRACT, HORIZON, EPS, encode_window, decode_targets
from mibot.data.datasets.orchardbench_dataset import load_stats

OUT = Path(__file__).resolve().parent
XR0 = OUT.parent
ORCHARD = Path('/home/rosmontis/Projects/orchardbench')
DATA = ORCHARD / 'data/orchard_v1_2650/filtered'
STATS = DATA / 'action_stats.json'
PRETRAINED = XR0.parent / 'checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D'
TRAINED = XR0 / 'outputs/orchard_v1_2650/project_orchardbench/orchard_v1_2650_frozen_vlm_seed42/epoch=0-step=10000.ckpt'
SELECTION = OUT / 'selection.json'
CHECKPOINTS = {'pretrained': PRETRAINED, 'step_10000': TRAINED}
SOURCE_FILES = [ORCHARD / 'treesim/orchard_action.py', ORCHARD / 'treesim/vla_env.py',
                XR0 / 'mibot/server/orchard_policy.py', XR0 / 'mibot/data/datasets/orchardbench_dataset.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    assert path.resolve().is_relative_to(OUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def intent_reference():
    """Reuse the existing method without importing the simulation/GPU modules."""
    path = ORCHARD / 'treesim/vla_env.py'
    tree = ast.parse(path.read_text())
    cfg = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'VLAEnvConfig')
    names = ('gripper_open_width', 'gripper_open_steps', 'gripper_close_deadband')
    constants = {x.target.id: ast.literal_eval(x.value) for x in cfg.body
                 if isinstance(x, ast.AnnAssign) and isinstance(x.target, ast.Name) and x.target.id in names}
    env = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'OrchardVLAEnv')
    method = copy.deepcopy(next(x for x in env.body if isinstance(x, ast.FunctionDef) and x.name == '_update_width_intent'))
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), 'exec'), ns)
    return constants, ns['_update_width_intent']


INTENT_CONSTANTS, UPDATE_INTENT = intent_reference()


def intent_state(widths, state=None):
    if state is None:
        state = dict(intent=0., open_steps=0, previous_width_command_m=.04)
    obj = SimpleNamespace(config=SimpleNamespace(**INTENT_CONSTANTS), _gripper=state['intent'],
                          _width_open_steps=state['open_steps'],
                          _gripper_width_command=state['previous_width_command_m'])
    intents, open_steps, previous = [], [], []
    for raw in widths:
        width = float(np.clip(raw, 0., .08))
        previous.append(obj._gripper_width_command)
        UPDATE_INTENT(obj, width)
        obj._gripper_width_command = width
        intents.append(obj._gripper)
        open_steps.append(obj._width_open_steps)
    final = dict(intent=float(obj._gripper), open_steps=int(obj._width_open_steps),
                 previous_width_command_m=float(obj._gripper_width_command))
    return np.asarray(intents), np.asarray(open_steps), np.asarray(previous), final


def phase_events(traj):
    times = np.asarray(traj['orchardbench']['timestamps'], dtype=float)
    events = []
    for event in traj['orchardbench']['fixed_base_expert']['state_trace']:
        # frame is physics 60 Hz; sim_time is rounded to 4 decimals in source.
        exact_time = event['frame'] / 60.
        assert abs(exact_time - event['sim_time']) <= .000051
        idx = int(np.searchsorted(times, exact_time - 1e-7, side='left'))
        events.append(dict(phase=event['state'], physics_frame=int(event['frame']),
                           event_time_s=exact_time, recorded_frame=min(idx, len(times) - 1),
                           mapped_time_s=float(times[min(idx, len(times) - 1)])))
    return events


def phases_at(traj, frame_indices):
    events = phase_events(traj)
    event_times = np.array([x['event_time_s'] for x in events])
    times = np.asarray(traj['orchardbench']['timestamps'])[frame_indices]
    indices = np.searchsorted(event_times, times + 1e-7, side='right') - 1
    return np.asarray([events[max(0, int(i))]['phase'] for i in indices])


def choose_windows(traj):
    events = phase_events(traj)
    starts = {x['phase']: x['recorded_frame'] for x in events}
    n = traj['num_frames']; last = n - HORIZON
    g, p, t, d = [starts[k] for k in ('GRASP', 'PULL', 'TRANSPORT', 'DROP')]
    candidates = [('reset', 0), ('before_grasp', g - 2), ('grasp_entry', g),
                  ('after_grasp_entry', g + 2), ('before_pull', p - 2), ('pull_entry', p),
                  ('after_pull_entry', p + 3), ('transport_entry', t),
                  ('transport_middle', (t + d) // 2), ('drop_entry_or_last_full_window', min(d, last))]
    result = []
    for label, requested in candidates:
        f = min(last, max(0, requested))
        if any(x['frame'] == f for x in result):
            raise ValueError('Fixed phase selection contains duplicate frames; inspect without model-based reselection')
        result.append(dict(label=label, frame=int(f), time_s=float(traj['orchardbench']['timestamps'][f]),
                           anchor_phase=str(phases_at(traj, np.array([f]))[0])))
    return result, events


def load_trajectory(ep):
    path = Path(ep['annotation'])
    if sha(path) != ep['annotation_sha256']:
        raise ValueError(f'Annotation changed since selection: {path}')
    return json.loads(path.read_text())


def make_selection():
    if SELECTION.exists():
        raise FileExistsError(f'Refuse to overwrite {SELECTION}; use --validate for existing selection')
    rows = [json.loads(s) for s in (DATA / 'manifest.jsonl').read_text().splitlines()]
    desired = {'train': [f'episode_{i:06d}' for i in (1, 2, 3, 4)],
               'val': [f'episode_{i:06d}' for i in (10, 20, 30, 40)]}
    episodes = []
    for split, ids in desired.items():
        for eid in ids:
            row = next(x for x in rows if x['episode_id'] == eid and x['split'] == split and x['accepted'])
            path = DATA / row['annotation']; traj = json.loads(path.read_text())
            windows, events = choose_windows(traj)
            episodes.append(dict(split=split, episode_id=eid, seed=traj['seed'], annotation=str(path),
                                 annotation_sha256=sha(path), num_frames=traj['num_frames'],
                                 events=events, windows=windows))
    selection = dict(schema='orchard_offline_probe_1003_v1', contract=CONTRACT, model_seed=42,
                     dataset_root=str(DATA), stats_path=str(STATS), stats_sha256=sha(STATS),
                     source_sha256={str(p): sha(p) for p in SOURCE_FILES},
                     selection_rule='Fixed episode IDs, then 10 deterministic phase windows; no model results consulted.',
                     phase_mapping='physics_frame/60 -> first recorded timestamps >= event_time - 1e-7; trace sim_time is rounded.',
                     intent_reference=dict(constants=INTENT_CONSTANTS,
                         source_method='treesim.vla_env.OrchardVLAEnv._update_width_intent (AST extracted unchanged)',
                         interpretation='Proxy from clipped expert measured-width targets, one target per 30Hz step; historical reference from episode reset. Not expert intent or reach-conditioned execution.'),
                     episodes=episodes)
    write_json(SELECTION, selection)
    return selection


def read_selection():
    s = json.loads(SELECTION.read_text())
    assert s['stats_sha256'] == sha(STATS)
    for path, digest in s['source_sha256'].items():
        assert sha(path) == digest, f'Source changed: {path}'
    return s


def observation(traj, frame, readers):
    p = traj['proprios']
    obs = dict(tcp_pos_world=np.asarray(p['ee_pos'][frame]),
               tcp_quat_world=Rotation.from_matrix(np.asarray(p['ee_rotm'][frame]).reshape(3, 3)).as_quat(),
               joint_pos=np.r_[p['arm_joint'][frame], 0., 0.],
               gripper_width=float(p['gripper_pos'][frame][0]))
    for name, key in (('rgb_static', 'ego'), ('rgb_wrist', 'wrist_left')):
        path = Path(traj['observations'][key][0]['path'])
        if not path.is_absolute(): path = DATA / path
        if not path.exists(): path = DATA / 'videos' / path.name
        if path not in readers:
            readers[path] = VideoReader(str(path), num_threads=1)
            assert len(readers[path]) == traj['num_frames']
        obs[name] = readers[path][frame].asnumpy()
        assert obs[name].dtype == np.uint8 and obs[name].shape == (144, 192, 3)
    return obs


def ground_truth(traj, frame):
    gt = encode_window(traj, frame)
    p = traj['proprios']; rotation = np.asarray(p['ee_rotm'][frame]).reshape(3, 3)
    world = decode_targets(gt, p['ee_pos'][frame], rotation)
    sl = slice(frame, frame + HORIZON)
    assert np.allclose(world[0], traj['actions']['ee_pos'][sl], atol=2e-7, rtol=0)
    assert np.allclose(world[1], np.asarray(traj['actions']['ee_rotm'][sl]).reshape(-1, 3, 3), atol=2e-7, rtol=0)
    assert np.allclose(world[2], np.asarray(traj['actions']['gripper_pos'][sl]).ravel(), atol=1e-8, rtol=0)
    return gt, world


def closure(widths, intents, previous, initial_intent):
    before = np.r_[initial_intent, intents[:-1]]
    switches = np.flatnonzero((intents < 0) & (before >= 0)) + 1
    slopes = np.flatnonzero(np.clip(widths, 0, .08) < previous - INTENT_CONSTANTS['gripper_close_deadband']) + 1
    closed = np.flatnonzero(intents < 0) + 1
    return dict(first_close_switch_horizon=None if len(switches) == 0 else int(switches[0]),
                first_close_intent_horizon=None if len(closed) == 0 else int(closed[0]),
                first_width_decrease_horizon=None if len(slopes) == 0 else int(slopes[0]),
                close_switch_horizons=switches.tolist(),
                initial_intent=float(initial_intent), width_out_of_bounds_count=int(np.sum((widths < 0) | (widths > .08))))


def evaluate_window(label, ep, win, traj, pred, normalized, output_dir):
    f = win['frame']; gt, gt_world = ground_truth(traj, f)
    p = traj['proprios']; anchor_r = np.asarray(p['ee_rotm'][f]).reshape(3, 3)
    pred_world = decode_targets(pred, p['ee_pos'][f], anchor_r)
    pos_error = np.linalg.norm(pred_world[0] - gt_world[0], axis=-1)
    rot_error = Rotation.from_matrix(gt_world[1].transpose(0, 2, 1) @ pred_world[1]).magnitude()
    width_error = np.abs(pred[:, 6] - gt[:, 6])
    _, _, _, history = intent_state(np.asarray(traj['actions']['gripper_pos'])[:f, 0])
    gi, go, gp, _ = intent_state(gt[:, 6], history)
    pi, po, pp, _ = intent_state(pred[:, 6], history)
    target_indices = np.minimum(np.arange(f + 1, f + HORIZON + 1), traj['num_frames'] - 1)
    phases = phases_at(traj, target_indices)
    target_times = np.asarray(traj['orchardbench']['timestamps'])[target_indices]
    path = output_dir / f"{ep['split']}_{ep['episode_id']}_frame{f:04d}.npz"
    with path.open('xb') as handle:
        np.savez_compressed(handle, normalized_prediction=normalized, predicted_action_physical=pred,
            gt_action_physical=gt, predicted_world_position=pred_world[0], gt_world_position=gt_world[0],
            predicted_world_rotation=pred_world[1], gt_world_rotation=gt_world[1],
            predicted_width_raw_m=pred[:, 6], predicted_width_clipped_m=np.clip(pred[:, 6], 0, .08),
            gt_width_m=gt[:, 6], position_error_m=pos_error, rotation_error_rad=rot_error,
            width_error_m=width_error, width_clipped_error_m=np.abs(np.clip(pred[:, 6], 0, .08) - np.clip(gt[:, 6], 0, .08)),
            gt_width_intent_proxy=gi, predicted_width_intent_proxy=pi,
            gt_open_steps_proxy=go, predicted_open_steps_proxy=po,
            target_phase=phases, target_time_s=target_times, horizon=np.arange(1, HORIZON + 1),
            anchor_position=np.asarray(p['ee_pos'][f]), anchor_rotation=anchor_r,
            anchor_width_m=np.array(p['gripper_pos'][f][0]))
    gcx = closure(gt[:, 6], gi, gp, history['intent']); pcx = closure(pred[:, 6], pi, pp, history['intent'])
    both = gcx['first_close_switch_horizon'] is not None and pcx['first_close_switch_horizon'] is not None
    row = dict(checkpoint=label, split=ep['split'], episode_id=ep['episode_id'], seed=ep['seed'],
               frame=f, label=win['label'], anchor_phase=win['anchor_phase'], npz=str(path),
               position_mae_m=float(pos_error.mean()), rotation_mae_rad=float(rot_error.mean()),
               width_mae_m=float(width_error.mean()), intent_mismatch_fraction=float(np.mean(gi != pi)),
               initial_intent_proxy=history, gt_closure=gcx, predicted_closure=pcx,
               close_switch_offset_steps=(pcx['first_close_switch_horizon'] - gcx['first_close_switch_horizon']) if both else None)
    return row


def aggregate(rows):
    groups = {}
    for row in rows:
        with np.load(row['npz']) as a:
            for split in ('all', row['split']):
                for name, mask in [('all', np.ones(HORIZON, bool)), ('h01', np.arange(HORIZON) == 0),
                    ('h01_05', np.arange(HORIZON) < 5), ('h06_15', (np.arange(HORIZON) >= 5) & (np.arange(HORIZON) < 15)),
                    ('h16_30', np.arange(HORIZON) >= 15)]:
                    key = f'{split}/horizon/{name}'
                    groups.setdefault(key, []).append(np.stack([a['position_error_m'][mask], a['rotation_error_rad'][mask], a['width_error_m'][mask]], axis=1))
                for phase in np.unique(a['target_phase']):
                    mask = a['target_phase'] == phase; key = f'{split}/target_phase/{phase}'
                    groups.setdefault(key, []).append(np.stack([a['position_error_m'][mask], a['rotation_error_rad'][mask], a['width_error_m'][mask]], axis=1))
    report = {}
    for name, values in groups.items():
        arr = np.concatenate(values)
        report[name] = dict(target_count=len(arr))
        for i, metric in enumerate(('position_error_m', 'rotation_error_rad', 'width_error_m')):
            report[name][metric] = dict(mean=float(arr[:, i].mean()), median=float(np.median(arr[:, i])), p90=float(np.quantile(arr[:, i], .9)), max=float(arr[:, i].max()))
    return report


def run_label(selection, label):
    output_dir = OUT / 'offline' / label
    if output_dir.exists(): raise FileExistsError(f'Refuse to overwrite {output_dir}')
    output_dir.mkdir(parents=True)
    _, mean, std = load_stats(STATS)
    policy = None; started = time.monotonic()
    if label != 'mean_action':
        from mibot.server.orchard_policy import OrchardPolicy
        policy = OrchardPolicy(str(CHECKPOINTS[label]), str(PRETRAINED), str(STATS), device='cuda:0')
    rows = []
    try:
        for ep in selection['episodes']:
            traj = load_trajectory(ep); readers = {}
            for win in ep['windows']:
                obs = observation(traj, win['frame'], readers)
                if label == 'mean_action':
                    normalized = np.zeros((HORIZON, 32), np.float32)
                    pred = mean.copy()
                else:
                    normalized = policy.predict(obs, seed=42)
                    pred = normalized * (std + EPS) + mean
                    pred[:, 7:] = 0
                assert np.isfinite(pred).all()
                row = evaluate_window(label, ep, win, traj, pred, normalized, output_dir)
                rows.append(row)
                print(json.dumps(dict(event='window_complete', checkpoint=label, episode=ep['episode_id'], frame=win['frame'], count=len(rows))), flush=True)
            readers.clear()
        report = dict(checkpoint=label, selection=str(SELECTION), selection_sha256=sha(SELECTION),
                      checkpoint_path=None if label == 'mean_action' else str(CHECKPOINTS[label]),
                      load_report=None if policy is None else policy.load_report,
                      model_seed=42, elapsed_s=time.monotonic() - started, windows=len(rows),
                      metric_definition='Position Euclidean metres; SO(3) geodesic radians; absolute total-width metres. Action before controller clipping; width clipping also retained in NPZ.',
                      limitation='Teacher observations only; overlapping selected windows, no statistical generalization estimate, no simulation execution, no proof of target selection or grasp success. Mean baseline is unconditional per-horizon training mean, not zero-motion.',
                      width_intent_reference=selection['intent_reference'], groups=aggregate(rows), window_metrics=rows)
        write_json(output_dir / 'report.json', report)
    finally:
        if policy is not None:
            del policy
            gc.collect()
            import torch
            torch.cuda.empty_cache()


def validate_cpu(selection):
    import torch
    _, mean, std = load_stats(STATS)
    count = 0; videos = set(); phase_counts = {}
    for ep in selection['episodes']:
        traj = load_trajectory(ep); readers = {}
        assert len(ep['windows']) == 10
        for win in ep['windows']:
            f = win['frame']; assert 0 <= f <= traj['num_frames'] - HORIZON
            obs = observation(traj, f, readers)
            gt, world = ground_truth(traj, f)
            assert np.allclose(((gt - mean) / (std + EPS)) * (std + EPS) + mean, gt, atol=2e-7)
            assert np.isfinite(obs['tcp_quat_world']).all()
            phase_counts[win['anchor_phase']] = phase_counts.get(win['anchor_phase'], 0) + 1
            count += 1
        videos.update(str(x) for x in readers); readers.clear()
    assert count == 80
    assert not torch.cuda.is_initialized(), 'CPU validation unexpectedly initialized CUDA'
    return dict(status='PASS', selected_episodes=8, full_windows=count, decoded_video_files=len(videos),
                windows_by_anchor_phase=phase_counts, cuda_initialized=False,
                contract_roundtrip='PASS', normalization_roundtrip='PASS', rgb_shape=[144, 192, 3],
                notes='No model built or GPU work started. Selection is fixed before checkpoint inference.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--validate', action='store_true')
    mode.add_argument('--infer', action='store_true')
    parser.add_argument('--checkpoints', nargs='+', choices=list(CHECKPOINTS), default=list(CHECKPOINTS))
    args = parser.parse_args()
    if args.prepare:
        selection = make_selection()
        report = validate_cpu(selection)
        write_json(OUT / 'offline_cpu_validation.json', report)
        run_label(selection, 'mean_action')
        print(json.dumps(report, indent=2))
    elif args.validate:
        print(json.dumps(validate_cpu(read_selection()), indent=2))
    else:
        selection = read_selection()
        for label in args.checkpoints:
            run_label(selection, label)


if __name__ == '__main__':
    main()
