"""Shared CPU contracts and fixed generation panels for the isolated 128-episode A/B experiment."""


import bootstrap


import argparse


import ast


from contextlib import contextmanager


import copy


import gc


import hashlib


import json


import os


from pathlib import Path


import random


import time


from types import SimpleNamespace


import numpy as np


import torch


from torch.utils.data import Subset


from scipy.spatial.transform import Rotation


from decord import VideoReader


from PIL import Image


from mibot.data.datasets.orchardbench_dataset import OrchardBenchDataset, load_stats, policy_messages


from mibot.data.datamodule.orchardbench_datamodule import OrchardBenchDataModule


from treesim.orchard_action import (CONTRACT, HORIZON, EPS, OrchardActionAdapter, action_mask,
                                   encode_window, decode_targets, prepare_rgb)


from checkpoint_io import (file_sha256, frozen_parameter_sha256, save_trainable_checkpoint,
                           load_trainable_overlay)


ROOT, XR0, ORCHARD = bootstrap.ROOT, bootstrap.XR0, bootstrap.ORCHARD


DATA = ORCHARD / 'data/orchard_v1_2650/filtered'


STATS = DATA / 'action_stats.json'


PROCESSOR = XR0.parent / 'checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D'


BASE = XR0 / 'outputs/orchard_v1_2650/project_orchardbench/orchard_v1_2650_frozen_vlm_seed42/epoch=0-step=10000.ckpt'


SELECTION = bootstrap.AB / 'selection.json'


CACHE = bootstrap.AB / 'eval_cache.pt'


STEPS = 8000


EVAL_STEPS = [0, 2000, 4000, 8000]


PREDICTION_SEEDS = [42, 43, 44]


TRAIN_SEED = 42


SAMPLER_SEED = 42


METRICS = ['position_error_m', 'rotation_error_rad', 'width_error_m']


BANDS = {'h1': slice(0, 1), 'h1_5': slice(0, 5), 'h6_15': slice(5, 15),
         'h16_30': slice(15, 30), 'full': slice(0, 30)}


SOURCE_FILES = [ROOT / name for name in ('bootstrap.py', 'checkpoint_io.py', 'ab_common.py', 'train_l.py', 'contact_loss.py', 'contact_model.py', 'prepare_l.py', 'cpu_loss_checks.py', 'config_l.json', 'evaluate_reference.py')] + [
    XR0 / 'mibot/models/VLA/XR0.py', XR0 / 'mibot/models/runner/orchard_runner.py',
    XR0 / 'mibot/models/runner/base_runner.py', XR0 / 'mibot/utils/orchard_checkpoint.py',
    XR0 / 'mibot/utils/model_utils.py', XR0 / 'mibot/utils/io.py',
    XR0 / 'mibot/data/collate/custom_collate.py', XR0 / 'mibot/data/datasets/orchardbench_dataset.py',
    XR0 / 'mibot/data/datamodule/orchardbench_datamodule.py',
    ORCHARD / 'treesim/orchard_action.py', ORCHARD / 'treesim/vla_env.py', XR0 / 'mibot/server/orchard_policy.py']


BEST_RULE = {'criterion': 'lexicographic, selected TRAIN panel only; VAL is report-only',
    'order': ['number_of_failed_proposed_train_fit_gates', 'train_reset_full_position_mae_m',
              'train_h1_5_position_p90_m'],
    'tie': 'keep earlier checkpoint',
    'baseline_eligible': True,
    'gates': {'h1_5_position_p90_m': .01, 'h1_5_rotation_p90_rad': .08,
              'h1_5_width_mae_m': .003, 'reset_full_position_mae_m': .03,
              'close_switch_abs_target_offset_max': 2, 'missing_gt_close_switch_predictions': 0},
    'note': 'These are fitting diagnostics, not task success. Width intent is a 30Hz target-sequence proxy.'}


def write_json(path, value, replace=False):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT): raise ValueError('Write outside A/B directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    if replace:
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('w') as f: json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
        os.replace(temporary, path)
    else:
        with path.open('x') as f: json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def extract_intent_method():
    tree = ast.parse((ORCHARD / 'treesim/vla_env.py').read_text())
    c = next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='VLAEnvConfig')
    names = ('gripper_open_width','gripper_open_steps','gripper_close_deadband')
    constants = {x.target.id:ast.literal_eval(x.value) for x in c.body
                 if isinstance(x,ast.AnnAssign) and isinstance(x.target,ast.Name) and x.target.id in names}
    e = next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='OrchardVLAEnv')
    fn = copy.deepcopy(next(x for x in e.body if isinstance(x,ast.FunctionDef) and x.name=='_update_width_intent'))
    ns = {}; exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), 'width_intent_reference','exec'),ns)
    return constants, ns['_update_width_intent']


INTENT_CONFIG, INTENT_UPDATE = extract_intent_method()


def width_intent(widths, start=None):
    if start is None: start = dict(intent=0., open_steps=0, previous_width_command_m=.04)
    obj = SimpleNamespace(config=SimpleNamespace(**INTENT_CONFIG), _gripper=start['intent'],
                          _width_open_steps=start['open_steps'], _gripper_width_command=start['previous_width_command_m'])
    intents=[]
    for raw in widths:
        w=float(np.clip(raw,0.,.08)); INTENT_UPDATE(obj,w); obj._gripper_width_command=w; intents.append(obj._gripper)
    end=dict(intent=float(obj._gripper), open_steps=int(obj._width_open_steps), previous_width_command_m=float(obj._gripper_width_command))
    return np.asarray(intents),end


def close_switch(intents, initial_intent):
    index=np.flatnonzero((intents<0)&(np.r_[initial_intent,intents[:-1]]>=0))
    return None if not len(index) else int(index[0]+1)


def phase_metadata(traj, frame):
    trace=traj['orchardbench']['fixed_base_expert']['state_trace']
    event_times=np.array([x['frame']/60. for x in trace])
    times=np.asarray(traj['orchardbench']['timestamps'])
    def labels(indices):
        pi=np.searchsorted(event_times,times[indices]+1e-7,side='right')-1
        return [trace[max(0,int(i))]['state'] for i in pi]
    target_indices=np.minimum(np.arange(frame+1,frame+31),len(times)-1)
    return labels([frame])[0],labels(target_indices),times[target_indices]


def cpu_deployment_parity(dm, sample, trajectory):
    f=sample['meta']['frame'];p=trajectory['proprios']
    obs=dict(tcp_pos_world=np.asarray(p['ee_pos'][f]),
        tcp_quat_world=Rotation.from_matrix(np.asarray(p['ee_rotm'][f]).reshape(3,3)).as_quat(),
        joint_pos=np.r_[p['arm_joint'][f],0.,0.], gripper_width=float(p['gripper_pos'][f][0]))
    images=[]
    for key in ('ego','wrist_left'):
        path=Path(trajectory['observations'][key][0]['path'])
        if not path.is_absolute():path=DATA/path
        if not path.exists():path=DATA/'videos'/path.name
        frame=VideoReader(str(path),num_threads=1)[f].asnumpy()
        images.append(prepare_rgb(Image.fromarray(frame)))
    deployment=dict(messages=policy_messages(images),state=torch.from_numpy(OrchardActionAdapter.state(obs)),
                    action=torch.zeros(30,32),action_mask=torch.from_numpy(action_mask()))
    actual=dm.collate_fn([deployment]);expected=dict(sample['batch']);expected['action']=torch.zeros_like(expected['action'])
    assert set(actual)==set(expected)
    details={}
    for k in actual:
        a,b=actual[k],expected[k]
        assert a.shape==b.shape and a.dtype==b.dtype
        exact=torch.equal(a,b)
        acceptable=exact or (k=='state' and torch.allclose(a,b,atol=2e-6,rtol=0))
        assert acceptable, f'Deployment tensor parity failed: {k}'
        details[k]=dict(shape=list(a.shape),dtype=str(a.dtype),exact_equal=exact,
                        maximum_difference=float((a.float()-b.float()).abs().max()))
    return details


def cpu_objective_and_overlay_test():
    from mibot.models.VLA.XR0 import XR0 as XR0Class
    fake=SimpleNamespace(freq_coefficient=0.)
    pred=torch.randn(1,30,32,requires_grad=True);target=torch.randn_like(pred);mask=torch.from_numpy(action_mask())[None]
    loss=XR0Class.compute_loss(fake,pred,target,mask)['loss']
    reference=.5*(pred[:,:,:7]-target[:,:,:7]).square().mean()
    assert torch.allclose(loss,reference,atol=1e-7)
    loss.backward();assert torch.count_nonzero(pred.grad[:,:,7:])==0
    # Reuse full loader on a tiny real torch module, verify frozen base untouched.
    fixture=torch.nn.Module();fixture.vlm=torch.nn.Linear(3,2);fixture.head=torch.nn.Linear(2,2)
    fixture.vlm.requires_grad_(False)
    frozen=frozen_parameter_sha256(fixture)
    base=ROOT/'tmp/cpu_overlay_base.bin'
    if not base.exists():base.write_bytes(b'A/B CPU overlay selftest base, not a model checkpoint\n')
    path=ROOT/'tmp/cpu_overlay_selftest.pt'
    desired={k:p.detach().clone() for k,p in fixture.named_parameters() if not k.startswith('vlm.')}
    saved=save_trainable_checkpoint(fixture,path,base_checkpoint=base,step=0,metadata={'test_only':True})
    with torch.no_grad():
        for n,p in fixture.named_parameters():
            if not n.startswith('vlm.'):p.add_(3.)
    loaded=load_trainable_overlay(fixture,path,base_checkpoint=base)
    assert all(torch.equal(dict(fixture.named_parameters())[k],v) for k,v in desired.items())
    assert frozen_parameter_sha256(fixture)==frozen
    return dict(loss_matches_production_active7_mean=True,inactive_gradient_zero=True,
                overlay_exact_parameter_restore=True,overlay_leaves_vlm_unchanged=True)


@contextmanager
def preserve_rng():
    py=random.getstate();npy=np.random.get_state();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:yield
    finally:
        random.setstate(py);np.random.set_state(npy);torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)


def to_device(batch,device,for_generation=False):
    # XR0.forward pops action/state/mask and casts tensors in this fresh dictionary.
    result={k:v.to(device,non_blocking=False) for k,v in batch.items()}
    if for_generation:result['action']=torch.zeros_like(result['action'])
    assert 'prefix_length' not in result
    return result


def dist(values):
    v=np.asarray(values,dtype=float).reshape(-1)
    return dict(count=len(v),mean=float(v.mean()),p50=float(np.median(v)),p90=float(np.quantile(v,.9)),max=float(v.max()))


def metrics_for_errors(errors):
    return {band:{metric:dist(errors[metric][:,sl]) for metric in METRICS} for band,sl in BANDS.items()}


def evaluate(model,samples,mean,std,step,device,output_root,status_callback=None,apply_fit_gate=True):
    final=Path(output_root)/'evaluations'/f'step_{step:04d}'
    if final.exists():
        report=json.loads((final/'report.json').read_text())
        assert report['step']==step and report['selection_sha256']==file_sha256(SELECTION)
        assert len(list(final.glob('*.npz')))==len(samples)
        return report
    directory=final.with_name(f'.step_{step:04d}.pending_{time.time_ns()}')
    directory.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();was_training=model.training;rows=[];group_errors={};target_phase_errors={}
    with preserve_rng(),torch.inference_mode():
        model.eval()
        try:
            for sample_index,sample in enumerate(samples):
                meta=sample['meta'];gt=sample['gt_action_physical'].numpy()
                gt_world=decode_targets(gt,meta['anchor_position'],meta['anchor_rotation'])
                results=[]
                # A new batch dict and zero-action input for every call; no prefix/GT leakage.
                for seed in PREDICTION_SEEDS:
                    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                    batch=to_device(sample['batch'],device,for_generation=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model.generate(batch)[0]
                    z=prediction.float().cpu().numpy();z[:,7:]=0
                    if not np.isfinite(z).all():raise FloatingPointError('Nonfinite generated action')
                    results.append(z)
                    del batch,prediction
                z=np.stack(results);physical=z*(std[None]+EPS)+mean[None]
                physical[:,:,7:]=0
                worlds=[decode_targets(a,meta['anchor_position'],meta['anchor_rotation']) for a in physical]
                pp=np.stack([w[0] for w in worlds]);pr=np.stack([w[1] for w in worlds])
                er=Rotation.from_matrix((gt_world[1][None].transpose(0,1,3,2)@pr).reshape(-1,3,3)).magnitude().reshape(3,30)
                errors=dict(position_error_m=np.linalg.norm(pp-gt_world[0][None],axis=-1),
                            rotation_error_rad=er,width_error_m=np.abs(physical[:,:,6]-gt[None,:,6]))
                for phase in sorted(set(meta['target_phase_by_horizon'])):
                    select=np.asarray(meta['target_phase_by_horizon'])==phase
                    bucket=target_phase_errors.setdefault(phase,{m:[] for m in METRICS})
                    for metric in METRICS:bucket[metric].append(errors[metric][:,select].reshape(-1))
                initial=meta['initial_width_intent_reference'];gi,_=width_intent(gt[:,6],initial)
                gt_switch=close_switch(gi,initial['intent']);pis=[];closure=[]
                for si,seed in enumerate(PREDICTION_SEEDS):
                    pi,_=width_intent(physical[si,:,6],initial);pis.append(pi)
                    switch=close_switch(pi,initial['intent'])
                    closure.append(dict(seed=seed,gt_first_close_switch_horizon=gt_switch,
                        predicted_first_close_switch_horizon=switch,
                        close_switch_offset_targets=None if switch is None or gt_switch is None else switch-gt_switch,
                        missing_gt_close_switch=gt_switch is not None and switch is None,
                        width_intent_mismatch_fraction=float(np.mean(pi!=gi))))
                path=directory/f"{meta['split']}_{meta['episode_id']}_frame{meta['frame']:04d}.npz"
                with path.open('xb') as f:
                    np.savez_compressed(f,prediction_seeds=np.array(PREDICTION_SEEDS),normalized_prediction=z,
                        predicted_action_physical=physical,gt_action_physical=gt,predicted_world_position=pp,
                        gt_world_position=gt_world[0],predicted_world_rotation=pr,gt_world_rotation=gt_world[1],
                        predicted_width_raw_m=physical[:,:,6],gt_width_m=gt[:,6],
                        predicted_width_intent_proxy=np.stack(pis),gt_width_intent_proxy=gi,
                        target_phase=np.array(meta['target_phase_by_horizon']),target_time_s=np.array(meta['target_times_s']),
                        horizon=np.arange(1,31),**errors)
                row={k:meta[k] for k in ('window_id','split','episode_id','seed','frame','label','labels','anchor_phase')}
                row.update(evaluation_set=meta['evaluation_set'],npz=str(final/path.name),metrics=metrics_for_errors(errors),closure_proxy=closure,
                    by_seed=[dict(seed=seed,metrics=metrics_for_errors({m:errors[m][si:si+1] for m in METRICS}))
                             for si,seed in enumerate(PREDICTION_SEEDS)])
                rows.append(row)
                group_names=['all',f"phase_{meta['anchor_phase']}",f"label_{meta['label']}"]
                if meta['frame']==0:group_names.append('reset')
                for split in ('all',meta['split'],meta['evaluation_set']):
                    for group in dict.fromkeys(group_names):
                        bucket=group_errors.setdefault(f'{split}/{group}',{m:[] for m in METRICS})
                        for metric in METRICS:bucket[metric].append(errors[metric])
                if (sample_index+1)%20==0:
                    if status_callback is not None:status_callback(sample_index+1,len(samples))
                    print(json.dumps(dict(event='eval_progress',step=step,windows=sample_index+1,total=len(samples))),flush=True)
        finally:
            model.train(was_training);model.vlm.eval()
    groups={key:metrics_for_errors({m:np.concatenate(v,axis=0) for m,v in bucket.items()}) for key,bucket in group_errors.items()}
    report=dict(schema='orchard_ab_generation_eval_v1',step=step,windows=len(rows),train_windows=sum(s['meta']['split']=='train' for s in samples),val_windows=sum(s['meta']['split']=='val' for s in samples),evaluation_sets={k:sum(s['meta']['evaluation_set']==k for s in samples) for k in sorted({s['meta']['evaluation_set'] for s in samples})},
        seeds=PREDICTION_SEEDS,seed_mode='Reset same seed independently for each window; training Python/numpy/CPU/CUDA RNG fully restored after panel.',
        generated_from_zero_action=True,ground_truth_prefix_used=False,model_num_steps=model.num_steps,
        by_actual_target_phase={phase:{metric:dist(np.concatenate(values)) for metric,values in bucket.items()} for phase,bucket in target_phase_errors.items()},
        actual_target_phase_note='Per-target phase, DROP and DONE remain separate; DONE width changes alone are not evidence of failed release.',
        elapsed_s=time.monotonic()-started,groups=groups,window_metrics=rows,
        metric_definition='Physical prediction errors before controller clipping; position Euclidean metres, SO3 geodesic radians, absolute total width metres. Groups pool targets and noise seeds, not independent episodes.',
        closure_reference='Original width-intent method on expert-target history and 30Hz predicted target sequence; not actual reach-conditioned execution timing.',
        selection_sha256=file_sha256(SELECTION),stats_sha256=file_sha256(STATS))
    report['fit_gate']=fit_gate(report) if apply_fit_gate else None
    report['used_for_checkpoint_selection']=bool(apply_fit_gate)
    write_json(directory/'report.json',report)
    os.replace(directory,final)
    print(json.dumps(dict(event='evaluation_complete',step=step,elapsed_s=report['elapsed_s'],fit_gate=report['fit_gate'])),flush=True)
    return report


def fit_gate(report):
    train=report['groups']['train/all'];reset=report['groups']['train/reset']
    closures=[c for w in report['window_metrics'] if w['split']=='train' for c in w['closure_proxy'] if c['gt_first_close_switch_horizon'] is not None]
    missing=sum(c['missing_gt_close_switch'] for c in closures)
    offsets=[abs(c['close_switch_offset_targets']) for c in closures if c['close_switch_offset_targets'] is not None]
    values=dict(h1_5_position_p90_m=train['h1_5']['position_error_m']['p90'],
        h1_5_rotation_p90_rad=train['h1_5']['rotation_error_rad']['p90'],
        h1_5_width_mae_m=train['h1_5']['width_error_m']['mean'],
        reset_full_position_mae_m=reset['full']['position_error_m']['mean'],
        close_switch_abs_target_offset_max=max(offsets) if offsets else None,
        missing_gt_close_switch_predictions=missing)
    passed={k:(values[k] is not None and values[k]<=threshold) for k,threshold in BEST_RULE['gates'].items()}
    if not closures:passed['close_switch_abs_target_offset_max']=False
    failed=sum(not x for x in passed.values())
    return dict(values=values,pass_by_gate=passed,failed_gates=failed,all_pass=failed==0,
                ranking=[failed,values['reset_full_position_mae_m'],values['h1_5_position_p90_m']],
                selection_basis='train fitting gates only; validation never used for checkpoint selection',gt_switch_predictions=len(closures))


