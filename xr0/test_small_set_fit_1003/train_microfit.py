#!/usr/bin/env python3
"""Isolated M0 microfit: original full-30 flow objective, fixed true-generation panels.

--prepare: full Dataset -> Subset, CPU input caching/parity/tests; no GPU/model.
--run: root-scheduled 2000 updates plus baseline/250/500/1000/2000 evaluations.
Production source files, datasets, original stats and checkpoints are read-only.
"""
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
SELECTION = ROOT / 'selection.json'
CACHE = ROOT / 'input_cache.pt'
STEPS = 2000
EVAL_STEPS = [0, 250, 500, 1000, 2000]
PREDICTION_SEEDS = [42, 43, 44]
TRAIN_SEED = 42
SAMPLER_SEED = 42
METRICS = ['position_error_m', 'rotation_error_rad', 'width_error_m']
BANDS = {'h1': slice(0, 1), 'h1_5': slice(0, 5), 'h6_15': slice(5, 15),
         'h16_30': slice(15, 30), 'full': slice(0, 30)}
SOURCE_FILES = [Path(__file__), ROOT / 'bootstrap.py', ROOT / 'checkpoint_io.py',
    XR0 / 'mibot/models/VLA/XR0.py', XR0 / 'mibot/models/runner/orchard_runner.py',
    XR0 / 'mibot/models/runner/base_runner.py', XR0 / 'mibot/utils/orchard_checkpoint.py',
    XR0 / 'mibot/utils/model_utils.py', XR0 / 'mibot/utils/io.py',
    XR0 / 'mibot/data/collate/custom_collate.py', XR0 / 'mibot/data/datasets/orchardbench_dataset.py',
    XR0 / 'mibot/data/datamodule/orchardbench_datamodule.py',
    ORCHARD / 'treesim/orchard_action.py', ORCHARD / 'treesim/vla_env.py']
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
    if not path.is_relative_to(ROOT): raise ValueError('Write outside M0 directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    if replace:
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('w') as f: json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
        os.replace(temporary, path)
    else:
        with path.open('x') as f: json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def config():
    return dict(schema='orchard_m0_config_v1', base_checkpoint=str(BASE), stats=str(STATS),
        processor_path=str(PROCESSOR), vlm_config_path=str(PROCESSOR / 'config.json'),
        max_updates=STEPS, eval_steps=EVAL_STEPS, batch_size=1, accumulate_grad_batches=1,
        train_seed=TRAIN_SEED, sampler_seed=SAMPLER_SEED, prediction_seeds=PREDICTION_SEEDS,
        precision='bf16 autocast, FP32 trainable master weights; no GradScaler',
        freeze_vlm=True, action_shape=[30,32], state_shape=[1,32], training_repeat=1,
        async_train=False, enable_freq=False, flow_sampling='beta', num_steps=5,
        optimizer=dict(type='AdamW', lr=1e-5, betas=[.9,.95], weight_decay=.1, eps=1e-8, foreach=False),
        scheduler=dict(type='ConstantLR', factor=1., total_iters=1), gradient_clip_norm=1.,
        objective='Unchanged production full-30 masked normalized flow velocity MSE, coefficient .5',
        sampling='Complete fixed 80 train windows, shuffled without replacement per cycle with independent CPU generator.',
        cache='CPU preprocessed/collated inputs only; no VLM feature/KV cache or model-output cache.',
        evaluation='All 80 train + 80 val windows, generation from zeros action and fixed seed per window; three independent calls with seeds42/43/44, no GT action/prefix supplied.',
        best_rule=BEST_RULE)


def source_hashes():
    return {str(p): file_sha256(p) for p in SOURCE_FILES}


def get_selection_windows():
    selection = json.loads(SELECTION.read_text())
    windows = []
    for ep in selection['episodes']:
        for win in ep['windows']:
            item = dict(win)
            item.update(split=ep['split'], episode_id=ep['episode_id'], seed=int(ep['seed']),
                        annotation=ep['annotation'], annotation_sha256=ep.get('annotation_sha256'))
            item['frame'] = int(item['frame'])
            item.setdefault('window_id', f"{item['split']}/{item['episode_id']}/frame{item['frame']:04d}")
            item.setdefault('label', item.get('labels', ['window'])[0])
            item.setdefault('labels', [item['label']])
            windows.append(item)
    assert len(windows) == 160 and len({x['window_id'] for x in windows}) == 160
    for split in ('train', 'val'):
        chosen = [x for x in windows if x['split'] == split]
        assert len(chosen) == 80 and len({x['episode_id'] for x in chosen}) == 8
        assert all(sum(y['episode_id'] == eid for y in chosen) == 10 for eid in {x['episode_id'] for x in chosen})
    assert not ({x['seed'] for x in windows if x['split']=='train'} & {x['seed'] for x in windows if x['split']=='val'})
    return selection, windows


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
    if not base.exists():base.write_bytes(b'M0 CPU overlay selftest base, not a model checkpoint\n')
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


def prepare():
    if CACHE.exists() or (ROOT/'cpu_validation.json').exists():
        raise FileExistsError('Prepared artifacts already exist; use --check-prepared rather than overwrite')
    selection,windows=get_selection_windows();_,mean,std=load_stats(STATS)
    dm=OrchardBenchDataModule(dict(processor_path=str(PROCESSOR)))
    cache_samples=[];stats_fingerprints={};dataset_lengths={};parity=None
    for split in ('train','val'):
        params=dict(train_datasets=dict(root=str(DATA),split=split,episode_ids=None,stats_path=str(STATS),action_length=30,batch_size=1))
        ds=OrchardBenchDataset(params)
        lookup={(str(Path(path).resolve()),frame):i for i,(path,frame) in enumerate(ds.samples)}
        selected=[w for w in windows if w['split']==split]
        indices=[lookup[(str(Path(w['annotation']).resolve()),w['frame'])] for w in selected]
        subset=Subset(ds,indices)
        dataset_lengths[split]=len(ds);stats_fingerprints[split]=ds.stats['source_sha256']
        for j,meta in enumerate(selected):
            meta=dict(meta);f=meta['frame'];p=Path(meta['annotation'])
            digest=file_sha256(p)
            if meta.get('annotation_sha256') is not None:assert meta['annotation_sha256']==digest
            meta['annotation_sha256']=digest
            traj=json.loads(p.read_text());assert traj['split']==split and traj['episode_id']==meta['episode_id']
            batch=dm.collate_fn([subset[j]])
            assert all(isinstance(v,torch.Tensor) and v.device.type=='cpu' for v in batch.values())
            assert batch['action'].shape==(1,30,32) and batch['action'].dtype==torch.float32
            gt=encode_window(traj,f);normalized=(gt-mean)/(std+EPS)
            assert np.array_equal(batch['action'][0].numpy(),normalized)
            assert torch.equal(batch['action_mask'][0],torch.from_numpy(action_mask()))
            anchor_phase,target_phases,target_times=phase_metadata(traj,f)
            meta.update(anchor_phase=anchor_phase,target_phase_by_horizon=target_phases,
                        target_times_s=target_times.tolist(),dataset_index=indices[j],
                        anchor_position=traj['proprios']['ee_pos'][f],anchor_rotation=np.asarray(traj['proprios']['ee_rotm'][f]).reshape(3,3).tolist())
            _,state=width_intent(np.asarray(traj['actions']['gripper_pos'])[:f,0])
            meta['initial_width_intent_reference']=state
            sample=dict(meta=meta,batch={k:v.contiguous() for k,v in batch.items()},gt_action_physical=torch.from_numpy(gt))
            if parity is None:parity=cpu_deployment_parity(dm,sample,traj)
            cache_samples.append(sample)
        del ds,subset,lookup
    assert len(cache_samples)==160 and not torch.cuda.is_initialized()
    tests=cpu_objective_and_overlay_test()
    provenance=dict(selection_sha256=file_sha256(SELECTION),stats_sha256=file_sha256(STATS),
        base_checkpoint=str(BASE),base_sha256=file_sha256(BASE),source_hashes=source_hashes(),
        normalization_source_fingerprint=stats_fingerprints,dataset_lengths=dataset_lengths)
    payload=dict(schema='orchard_m0_cpu_input_cache_v1',provenance=provenance,samples=cache_samples,
                 mean=torch.from_numpy(mean),std=torch.from_numpy(std))
    with CACHE.open('xb') as f:torch.save(payload,f)
    report=dict(status='PASS',windows=160,train_windows=80,val_windows=80,
        dataset_lengths=dataset_lengths,full_dataset_then_subset=True,validation_uses_train_stats=True,
        dataset_to_deployment_tensor_parity=parity,tests=tests,cuda_initialized=torch.cuda.is_initialized(),
        input_cache_path=str(CACHE),input_cache_sha256=file_sha256(CACHE),input_cache_bytes=CACHE.stat().st_size,
        source_hashes=provenance['source_hashes'],selection_sha256=provenance['selection_sha256'],
        stats_sha256=provenance['stats_sha256'],base_sha256=provenance['base_sha256'])
    assert report['cuda_initialized'] is False
    write_json(ROOT/'cpu_validation.json',report)
    write_json(ROOT/'config.json',config())
    print(json.dumps(report,indent=2),flush=True)


def load_cache(check_sources=True):
    report=json.loads((ROOT/'cpu_validation.json').read_text())
    assert report['status']=='PASS' and report['input_cache_sha256']==file_sha256(CACHE)
    payload=torch.load(CACHE,map_location='cpu',weights_only=True,mmap=True)
    p=payload['provenance']
    assert p['selection_sha256']==file_sha256(SELECTION) and p['stats_sha256']==file_sha256(STATS)
    assert p['base_sha256']==file_sha256(BASE)
    if check_sources:
        for path,expected in p['source_hashes'].items():
            assert file_sha256(path)==expected, f'Source changed after prepare: {path}'
    for ep in json.loads(SELECTION.read_text())['episodes']:
        path=Path(ep['annotation']);known={x['meta']['annotation_sha256'] for x in payload['samples'] if x['meta']['episode_id']==ep['episode_id']}
        assert known=={file_sha256(path)}
    return payload


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


def evaluate(model,samples,mean,std,step,device):
    directory=ROOT/'evaluations'/f'step_{step:04d}'
    directory.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();was_training=model.training;rows=[];group_errors={}
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
                row.update(npz=str(path),metrics=metrics_for_errors(errors),closure_proxy=closure,
                    by_seed=[dict(seed=seed,metrics=metrics_for_errors({m:errors[m][si:si+1] for m in METRICS}))
                             for si,seed in enumerate(PREDICTION_SEEDS)])
                rows.append(row)
                group_names=['all',f"phase_{meta['anchor_phase']}",f"label_{meta['label']}"]
                if meta['frame']==0:group_names.append('reset')
                for split in ('all',meta['split']):
                    for group in dict.fromkeys(group_names):
                        bucket=group_errors.setdefault(f'{split}/{group}',{m:[] for m in METRICS})
                        for metric in METRICS:bucket[metric].append(errors[metric])
                if (sample_index+1)%20==0:
                    print(json.dumps(dict(event='eval_progress',step=step,windows=sample_index+1,total=len(samples))),flush=True)
        finally:
            model.train(was_training);model.vlm.eval()
    groups={key:metrics_for_errors({m:np.concatenate(v,axis=0) for m,v in bucket.items()}) for key,bucket in group_errors.items()}
    report=dict(schema='orchard_m0_generation_eval_v1',step=step,windows=len(rows),train_windows=80,val_windows=80,
        seeds=PREDICTION_SEEDS,seed_mode='Reset same seed independently for each window; training Python/numpy/CPU/CUDA RNG fully restored after panel.',
        generated_from_zero_action=True,ground_truth_prefix_used=False,model_num_steps=model.num_steps,
        elapsed_s=time.monotonic()-started,groups=groups,window_metrics=rows,
        metric_definition='Physical prediction errors before controller clipping; position Euclidean metres, SO3 geodesic radians, absolute total width metres. Groups pool targets and noise seeds, not independent episodes.',
        closure_reference='Original width-intent method on expert-target history and 30Hz predicted target sequence; not actual reach-conditioned execution timing.',
        selection_sha256=file_sha256(SELECTION),stats_sha256=file_sha256(STATS))
    report['fit_gate']=fit_gate(report)
    write_json(directory/'report.json',report)
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


def run():
    if (ROOT/'run_manifest.json').exists():raise FileExistsError('Existing M0 run; refuse overwrite/resume')
    payload=load_cache();samples=payload['samples'];mean=payload['mean'].numpy();std=payload['std'].numpy()
    train=[s for s in samples if s['meta']['split']=='train'];assert len(train)==80
    if not torch.cuda.is_available():raise RuntimeError('M0 --run requires the root-scheduled GPU')
    torch.set_num_threads(1)
    random.seed(TRAIN_SEED);np.random.seed(TRAIN_SEED);torch.manual_seed(TRAIN_SEED);torch.cuda.manual_seed_all(TRAIN_SEED)
    from mibot.models.runner.orchard_runner import OrchardRunner
    from mibot.models.runner.base_runner import BaseRunner
    params=dict(pretrained=str(BASE),freeze_vlm=True,diagnostic_path=None,
        model=dict(type='XR0',vlm_config_path=str(PROCESSOR/'config.json'),training_repeat=1,enable_freq=False,async_train=False))
    runner=OrchardRunner(params);runner.configure_model();model=runner.model
    assert model.num_steps==5 and model.flow_sampling=='beta' and model.training_repeat==1 and not model.async_train
    assert all(not p.requires_grad for p in model.vlm.parameters())
    assert all(p.dtype==torch.float32 for p in model.parameters() if p.requires_grad)
    frozen_before=frozen_parameter_sha256(model)
    initial_output=model.action_output_layer.layers[2].weight.detach().cpu().clone()
    model.to('cuda:0');device=torch.device('cuda:0');model.train();model.vlm.eval()
    optimizer_cfg=dict(type='torch.optim.AdamW',params=dict(lr=1e-5,betas=(.9,.95),weight_decay=.1,eps=1e-8,foreach=False))
    # Prefix names with model. exactly as BaseRunner.named_parameters() does.
    optimizer=BaseRunner.build_optimizer(optimizer_cfg,runner.named_parameters())
    scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer,factor=1.,total_iters=1)
    assert len(optimizer.state)==0
    sampler=torch.Generator(device='cpu');sampler.manual_seed(SAMPLER_SEED)
    # Explicit stochastic start after model construction. Evaluation preserves this stream.
    random.seed(TRAIN_SEED);np.random.seed(TRAIN_SEED);torch.manual_seed(TRAIN_SEED);torch.cuda.manual_seed_all(TRAIN_SEED)
    manifest=dict(schema='orchard_m0_run_v1',config=config(),provenance=payload['provenance'],
        current_source_hashes=source_hashes(),input_cache_sha256=file_sha256(CACHE),
        initial_load_report=runner.evidence,initial_optimizer_state_entries=len(optimizer.state),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_vlm_parameter_sha256_before=frozen_before,gpu_name=torch.cuda.get_device_name(0),
        torch_version=str(torch.__version__),source_unchanged_at_run_start=True,
        best_rule=BEST_RULE,training_windows=[s['meta']['window_id'] for s in train],
        validation_windows=[s['meta']['window_id'] for s in samples if s['meta']['split']=='val'],
        checkpoint_format='trainable-only FP32 overlay; verified original 10k base required; no optimizer state serialized')
    write_json(ROOT/'run_manifest.json',manifest)
    trainable=[p for p in model.parameters() if p.requires_grad]
    counts={s['meta']['window_id']:0 for s in train};best_rank=None;best_record=None;reports=[];order=[];cursor=0
    started=time.monotonic();training_seconds=0.;evaluation_seconds=0.
    metadata=dict(selection_sha256=file_sha256(SELECTION),stats_sha256=file_sha256(STATS),
        source_hashes=source_hashes(),config=config(),frozen_vlm_parameter_sha256=frozen_before)
    baseline=evaluate(model,samples,mean,std,0,device);reports.append(dict(step=0,path=str(ROOT/'evaluations/step_0000/report.json'),fit_gate=baseline['fit_gate']))
    evaluation_seconds+=baseline['elapsed_s'];best_rank=tuple(baseline['fit_gate']['ranking'])
    best_record=save_trainable_checkpoint(model,ROOT/'checkpoints/best_trainable.pt',base_checkpoint=BASE,step=0,
                                         metadata={**metadata,'fit_gate':baseline['fit_gate'],'selection_rule':BEST_RULE})
    write_json(ROOT/'best_checkpoint.json',{**best_record,'fit_gate':baseline['fit_gate'],'rule':BEST_RULE})
    with (ROOT/'training_steps.jsonl').open('x') as log:
        for step in range(1,STEPS+1):
            if cursor>=len(order):order=torch.randperm(len(train),generator=sampler).tolist();cursor=0
            index=order[cursor];cursor+=1;sample=train[index];meta=sample['meta'];assert meta['split']=='train'
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize();tic=time.monotonic()
            batch=to_device(sample['batch'],device)
            with torch.autocast('cuda',dtype=torch.bfloat16):losses=model(batch,return_loss=True)
            loss=losses['loss']
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite train loss at {step}')
            loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();torch.cuda.synchronize()
            seconds=time.monotonic()-tic;training_seconds+=seconds;counts[meta['window_id']]+=1
            item=dict(step=step,window_id=meta['window_id'],split='train',episode_id=meta['episode_id'],frame=meta['frame'],
                label=meta['label'],anchor_phase=meta['anchor_phase'],flow_loss=float(loss.detach()),
                flow_loss_mse=float(losses['loss_mse'].detach()),flow_loss_freq=float(losses['loss_freq'].detach()),
                gradient_norm_before_clip=float(grad_norm.detach()),gradients_finite=True,
                lr=float(optimizer.param_groups[0]['lr']),step_seconds=seconds,
                train_seconds_cumulative=training_seconds,optimizer_state_entries=len(optimizer.state))
            log.write(json.dumps(item,allow_nan=False)+'\n')
            if step%25==0:
                log.flush();print(json.dumps(dict(event='train_progress',**item)),flush=True)
            del batch,losses,loss,grad_norm
            if step in EVAL_STEPS:
                optimizer.zero_grad(set_to_none=True)
                result=evaluate(model,samples,mean,std,step,device)
                evaluation_seconds+=result['elapsed_s'];rank=tuple(result['fit_gate']['ranking'])
                reports.append(dict(step=step,path=str(ROOT/'evaluations'/f'step_{step:04d}'/'report.json'),fit_gate=result['fit_gate']))
                if rank<best_rank:
                    best_rank=rank
                    best_record=save_trainable_checkpoint(model,ROOT/'checkpoints/best_trainable.pt',base_checkpoint=BASE,step=step,
                        metadata={**metadata,'fit_gate':result['fit_gate'],'selection_rule':BEST_RULE})
                    write_json(ROOT/'best_checkpoint.json',{**best_record,'fit_gate':result['fit_gate'],'rule':BEST_RULE},replace=True)
                write_json(ROOT/'training_progress.json',dict(completed_updates=step,best_step=best_record['step'],
                    per_window_counts=counts,unique_train_windows_seen=sum(v>0 for v in counts.values()),
                    val_optimizer_updates=0,evaluations=reports),replace=True)
    optimizer.zero_grad(set_to_none=True)
    frozen_after=frozen_parameter_sha256(model)
    if frozen_before!=frozen_after:raise AssertionError('Frozen VLM parameter bytes changed')
    update=float((model.action_output_layer.layers[2].weight.detach().cpu()-initial_output).abs().max())
    assert update>0 and sum(counts.values())==2000 and len(counts)==80 and all(v>0 for v in counts.values())
    assert all(not p.requires_grad and p.grad is None for p in model.vlm.parameters())
    final=save_trainable_checkpoint(model,ROOT/'checkpoints/final_trainable.pt',base_checkpoint=BASE,step=STEPS,
        metadata={**metadata,'fit_gate':reports[-1]['fit_gate'],'selection_rule':BEST_RULE})
    # Validate final overlay reload exactly on trainable parameters, without a second full model allocation.
    reload_report=load_trainable_overlay(model,ROOT/'checkpoints/final_trainable.pt',base_checkpoint=BASE)
    summary=dict(status='COMPLETE',completed_updates=STEPS,train_window_presentations=sum(counts.values()),
        unique_train_windows_seen=len(counts),val_optimizer_updates=0,per_window_counts=counts,
        frozen_vlm_parameter_sha256_before=frozen_before,frozen_vlm_parameter_sha256_after=frozen_after,
        frozen_vlm_unchanged=frozen_before==frozen_after,trainable_master_dtype='float32',
        output_weight_max_absolute_update=update,all_logged_gradients_finite=True,
        training_seconds=training_seconds,evaluation_seconds=evaluation_seconds,total_seconds=time.monotonic()-started,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        best_checkpoint=best_record,final_checkpoint=final,final_overlay_reload=reload_report,evaluations=reports,
        current_source_hashes=source_hashes())
    assert summary['current_source_hashes']==manifest['current_source_hashes']
    write_json(ROOT/'training_summary.json',summary)
    print(json.dumps(dict(event='training_complete',summary=str(ROOT/'training_summary.json'),best_step=best_record['step'],
                         final_step=STEPS,output_weight_max_update=update,frozen_vlm_unchanged=True)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--check-prepared',action='store_true');mode.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.check_prepared:
        cache=load_cache();assert not torch.cuda.is_initialized()
        print(json.dumps(dict(status='PASS',prepared_windows=len(cache['samples']),cuda_initialized=False),indent=2))
    else:run()


if __name__=='__main__':main()
