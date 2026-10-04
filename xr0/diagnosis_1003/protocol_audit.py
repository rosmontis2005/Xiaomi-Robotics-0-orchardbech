"""Read-only CPU diagnosis. Writes only protocol_audit.json beside this file.

No simulator/policy/model is constructed. Width intent calls the unchanged AST
node from vla_env.py, isolated from imports that could initialize the simulator.
"""
import bootstrap  # diagnostic-only path/cache setup
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

OUT = Path(__file__).resolve().parent
XR0 = OUT.parent
ORCHARD = Path('/home/rosmontis/Projects/orchardbench')
EXP = XR0 / 'outputs/orchard_v1_2650/project_orchardbench/orchard_v1_2650_frozen_vlm_seed42'
BENCH = XR0 / 'outputs/orchard_v1_benchmark'
FILTER = ORCHARD / 'log/v1_2650_collection_replay_filter'
DATA = ORCHARD / 'data/orchard_v1_2650/filtered'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p10=float(np.quantile(a, .1)), p90=float(np.quantile(a, .9)),
                minimum=float(a.min()), maximum=float(a.max())) if len(a) else dict(n=0)


def width_audit(env_path):
    source = env_path.read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OrchardVLAEnv')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_update_width_intent')
    isolated = ast.Module(body=[node], type_ignores=[])
    namespace = {}
    exec(compile(isolated, str(env_path), 'exec'), namespace)
    update = namespace['_update_width_intent']
    config = SimpleNamespace(gripper_open_width=.075, gripper_open_steps=5, gripper_close_deadband=.001)
    cases = [
        ('smooth_close_0p5mm_from_open', 1., .08, 5, [.08 - i*.0005 for i in range(1, 121)]),
        ('abrupt_close_2mm_from_open', 1., .08, 5, [.078, .076, .074, .072, .070, .060, .040, .020]),
        ('held_rebound_four_open_samples', -1., .032, 0, [.060, .079, .079, .079, .079, .032]),
        ('held_rebound_five_open_samples', -1., .032, 0, [.079]*5 + [.032]),
        ('repeated_open_after_close', -1., .020, 0, [.080]*6),
        ('stationary_narrow_from_neutral', 0., .040, 0, [.040]*6),
    ]
    results = []
    for name, intent, width, open_steps, widths in cases:
        obj = SimpleNamespace(config=config, _gripper=float(intent),
                              _gripper_width_command=float(width), _width_open_steps=open_steps)
        rows = []
        for i, requested in enumerate(widths):
            applied = min(.08, max(0., requested))
            prior = obj._gripper_width_command
            prior_intent = obj._gripper
            update(obj, applied)
            # This mirrors vla_env.step lines after _update_width_intent: set q[-2:]
            # to width / 2, then remember their sum. No fingers or contacts simulated.
            obj._gripper_width_command = applied
            rows.append(dict(step=i, requested_width_m=requested, applied_width_m=applied,
                             previous_command_width_m=prior, command_decrease_m=prior-applied,
                             intent_before=prior_intent, intent_after=obj._gripper,
                             open_streak_after=obj._width_open_steps))
        results.append(dict(name=name, initial_intent=intent, initial_width_m=width,
                            ever_close_intent=any(r['intent_after'] < 0 for r in rows),
                            final_intent=obj._gripper, final_width_m=obj._gripper_width_command,
                            rows=rows))
    return dict(method='AST extracted unchanged _update_width_intent; no simulator module import',
                source_path=str(env_path), source_sha256=sha(env_path),
                method_first_line=node.lineno, method_last_line=node.end_lineno,
                method_source=ast.get_source_segment(source, node),
                configuration=vars(config), scenarios=results,
                limits='Isolated intent state-machine behavior only; no evidence that any model produced these sequences, and no physical contact/grasp is simulated.')


def wandb_audit(path):
    from wandb.proto.wandb_internal_pb2 import Record
    from wandb.sdk.internal.datastore import DataStore
    # open_for_scan uses r+b. Bind a read-only handle to the same scanner instead.
    ds = DataStore()
    ds._fp = path.open('rb')
    ds._size_bytes = path.stat().st_size
    ds._opened_for_scan = True
    ds._read_header()
    kinds = Counter()
    history = []
    run_data = []
    try:
        while (raw := ds.scan_data()) is not None:
            record = Record()
            record.ParseFromString(raw)
            kind = record.WhichOneof('record_type')
            kinds[kind] += 1
            if kind == 'history':
                row = {}
                for item in record.history.item:
                    key = item.key if item.key else '/'.join(item.nested_key)
                    row[key] = json.loads(item.value_json)
                history.append(row)
            elif kind == 'run':
                run_data.append(dict(run_id=record.run.run_id, display_name=record.run.display_name,
                                     project=record.run.project, entity=record.run.entity))
    finally:
        ds._fp.close()
    loss_rows = [r for r in history if 'train/loss' in r]
    block_summaries = []
    for start in range(0, 10000, 1000):
        rows = [r for r in loss_rows if start <= r.get('trainer/global_step', -1) < start+1000]
        block_summaries.append(dict(global_step_lower_inclusive=start, global_step_upper_exclusive=start+1000,
                                   loss=stats([r['train/loss'] for r in rows]),
                                   loss_mse=stats([r['train/loss_mse'] for r in rows if 'train/loss_mse' in r])))
    return dict(path=str(path), sha256=sha(path), open_mode='rb', record_counts=dict(kinds),
                run_metadata=run_data, history_count=len(history), loss_count=len(loss_rows),
                global_step_range=[min(r['trainer/global_step'] for r in loss_rows),
                                   max(r['trainer/global_step'] for r in loss_rows)],
                lr_values=sorted({r['lr'] for r in history if 'lr' in r}),
                epoch_values=sorted({r['epoch'] for r in history if 'epoch' in r}),
                loss_overall=stats([r['train/loss'] for r in loss_rows]),
                loss_blocks_1000_steps=block_summaries, history=history,
                interpretation='Logged training batches only, sampled by logging cadence; no validation loss or convergence claim. No W&B run initialized or synced.')


def selected_gt(adapter_module, normalization):
    wanted = {2010027, 2010042, 2010067}
    answer = []
    mean, std = np.asarray(normalization['mean']), np.asarray(normalization['std'])
    for line_no, line in enumerate((FILTER/'v1_trainable_manifest.jsonl').read_text().splitlines(), 1):
        r = json.loads(line)
        if r['seed'] not in wanted:
            continue
        path = DATA / r['annotation']
        traj = load(path)
        frame = 0
        actions = adapter_module.encode_window(traj, frame)
        normalized = (actions-mean)/(std+adapter_module.EPS)
        rotation = np.asarray(traj['proprios']['ee_rotm'][frame]).reshape(3, 3)
        anchor = dict(tcp_pos_world=np.asarray(traj['proprios']['ee_pos'][frame]),
                      tcp_quat_world=Rotation.from_matrix(rotation).as_quat())
        adapter = adapter_module.OrchardActionAdapter()
        adapter.set_chunk(normalized, mean, std, anchor)
        xyz, rot, width = adapter.targets
        expected_xyz = np.asarray(traj['actions']['ee_pos'][:30])
        expected_rot = np.asarray(traj['actions']['ee_rotm'][:30]).reshape(30,3,3)
        expected_width = np.asarray(traj['actions']['gripper_pos'][:30]).reshape(30)
        xyz_before, rot_before, width_before = [v.copy() for v in adapter.targets]
        moved = dict(tcp_pos_world=anchor['tcp_pos_world']+np.array([.01, -.02, .03]),
                     tcp_quat_world=(Rotation.from_rotvec([.02, -.01, .03])*Rotation.from_quat(anchor['tcp_quat_world'])).as_quat())
        command = adapter.to_native(0, moved)
        expected_delta_rotation = rot[0]@Rotation.from_quat(moved['tcp_quat_world']).as_matrix().T
        command_rotation = Rotation.from_euler('xyz',command['action'][3:6]).as_matrix()
        answer.append(dict(seed=r['seed'], episode_id=r['episode_id'], split=r['split'],
                           manifest_line=line_no, annotation=str(path), resolved_annotation=str(path.resolve()),
                           annotation_exists=path.exists(), annotation_sha256=sha(path),
                           annotation_manifest_sha256=r['annotation_sha256'],
                           annotation_matches_manifest=sha(path)==r['annotation_sha256'],
                           num_frames=traj['num_frames'], oracle_result=r['oracle_result'],
                           oracle_strict_success=r['oracle_strict_success'], replay=r['replay'],
                           frame0_roundtrip=dict(max_position_error_m=float(np.max(np.linalg.norm(xyz-expected_xyz,axis=1))),
                                max_rotation_error_rad=float(np.max(Rotation.from_matrix(rot@np.transpose(expected_rot,(0,2,1))).magnitude())),
                                max_width_error_m=float(np.max(np.abs(width-expected_width)))),
                           live_observation_command_probe=dict(
                                targets_unchanged_after_to_native=all(np.array_equal(a,b) for a,b in zip(adapter.targets,(xyz_before,rot_before,width_before))),
                                translation_residual_m=float(np.linalg.norm(command['action'][:3]-(xyz[0]-moved['tcp_pos_world']))),
                                rotation_residual_rad=float(Rotation.from_matrix(command_rotation@expected_delta_rotation.T).magnitude()),
                                absolute_width_residual_m=float(abs(command['gripper_width']-width[0])))))
    assert len(answer)==3
    return answer


def main():
    output = OUT/'protocol_audit.json'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing audit: {output}')
    config_path = EXP/'config.yaml'
    stats_path = DATA/'action_stats.json'
    action_path = ORCHARD/'treesim/orchard_action.py'
    env_path = ORCHARD/'treesim/vla_env.py'
    original_paths = [config_path, stats_path, action_path, env_path,
                      XR0/'mibot/server/orchard_policy.py', XR0/'tools/eval_orchard_learning_curve.py',
                      FILTER/'sources/frozen_gate1_oracle.py', ORCHARD/'scripts/check_gate1_oracle.py']
    before = {str(p):sha(p) for p in original_paths}
    config = yaml.safe_load(config_path.read_text())
    normalization = load(stats_path)
    manifest = load(BENCH/'checkpoint_manifest.json')
    protocol = load(FILTER/'protocol.json')
    spec = importlib.util.spec_from_file_location('diagnosis_orchard_action',action_path)
    action = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(action)
    result = dict(schema='orchard_diagnosis_protocol_audit_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Read-only CPU audit; no model, simulator, new rollout, optimizer or network',
        source_hashes=before, actual_training_config_path=str(config_path), actual_training_config=config,
        stats=dict(path=str(stats_path), sha256=sha(stats_path),
                   matches_benchmark=sha(stats_path)==manifest['config']['stats_sha256'],
                   metadata={k:v for k,v in normalization.items() if k not in ('mean','std','episodes')},
                   episode_count=len(normalization['episodes']), mean_shape=np.asarray(normalization['mean']).shape,
                   std_shape=np.asarray(normalization['std']).shape),
        execution=dict(benchmark=manifest['config']['execution'], environment=manifest['config']['environment'],
                       frozen_replay_sha256=protocol['replay_source_sha256'],
                       frozen_source_matches=sha(FILTER/'sources/frozen_gate1_oracle.py')==protocol['replay_source_sha256'],
                       frozen_advancement='post-step position <= .01 m AND rotation <= .08 rad OR dwell >= 30; one target per advancement; no width gate',
                       current_gate1_has_passed_branch='or passed or expired' in (ORCHARD/'scripts/check_gate1_oracle.py').read_text(),
                       benchmark_selection=load(BENCH/'benchmark_v1_30seeds.json')['selection'],
                       anchor='model: current observation once at predict/set_chunk; replay: original expert anchor per encoded chunk; to_native only recomputes live-relative command, not anchor',
                       gripper_contract='action index 6 is absolute total finger opening in metres; native action index 6 is 0 and width travels separately as gripper_width',
                       clip_flag_limits='command, workspace and width clipping included; joint slew max_joint_step not included in action_clipped flag',
                       budget='900 env control steps, independent of target/chunk index; 30Hz control; 60Hz physics'),
        selected_gt=selected_gt(action, normalization),
        width_intent=width_audit(env_path),
        wandb=wandb_audit(XR0/'wandb/offline-run-20261002_102310-6ii6ejsj/run-6ii6ejsj.wandb'))
    after={str(p):sha(p) for p in original_paths}
    result['original_sources_unchanged']=before==after
    assert before==after
    with output.open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,default=jsonable)
        f.write('\n')
    print(json.dumps(dict(output=str(output),sources_unchanged=result['original_sources_unchanged'],
                         selected_gt=[r['seed'] for r in result['selected_gt']],
                         wandb_loss_rows=result['wandb']['loss_count'],
                         loss_blocks=[dict(step=b['global_step_lower_inclusive'],mean=b['loss'].get('mean')) for b in result['wandb']['loss_blocks_1000_steps']],
                         width_results=[dict(name=r['name'],final_intent=r['final_intent'],ever_close_intent=r['ever_close_intent']) for r in result['width_intent']['scenarios']]),ensure_ascii=False))


if __name__ == '__main__':
    main()
