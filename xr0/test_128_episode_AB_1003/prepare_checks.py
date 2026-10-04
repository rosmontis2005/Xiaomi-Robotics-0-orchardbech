#!/usr/bin/env python3
"""CPU-only schedule/leakage/contract/RNG/source-precision/resume checks.

No full VLA model is constructed and CUDA must stay uninitialized.
--refresh may replace preparation before a run; never while either arm exists.
"""
import bootstrap
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
import ab_common as common
import train_ab as training
from checkpoint_io import file_sha256
ROOT=bootstrap.ROOT


def cpu_runtime_smoke():
    assert not torch.cuda.is_initialized()
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    state=training.capture_rng()
    expected=(random.random(),np.random.rand(),torch.randn(3))
    training.restore_rng(state)
    with common.preserve_rng():
        random.seed(999);np.random.seed(999);torch.manual_seed(999)
        random.random();np.random.rand();torch.randn(3)
    actual=(random.random(),np.random.rand(),torch.randn(3))
    assert expected[0]==actual[0] and expected[1]==actual[1] and torch.equal(expected[2],actual[2])
    production=common.cpu_objective_and_overlay_test()
    # Exercise actual shared strict source loader on FP32 values not representable in BF16.
    from mibot.utils.orchard_checkpoint import load_weights
    fixture=torch.nn.Module();fixture.vlm=torch.nn.Linear(3,3).bfloat16();fixture.head=torch.nn.Linear(3,1).bfloat16()
    source={k:v.clone() for k,v in fixture.state_dict().items()}
    source['head.weight']=torch.tensor([[.10001,.20003,.30007]],dtype=torch.float32)
    source['head.bias']=torch.tensor([.01234567],dtype=torch.float32)
    p=ROOT/'tmp/source_precision_cpu_fixture.pt';torch.save({'state_dict':source},p)
    load_weights(fixture,p)
    assert fixture.head.weight.dtype==torch.float32 and torch.equal(fixture.head.weight,source['head.weight'])
    assert fixture.vlm.weight.dtype==torch.bfloat16
    # Verify exact Adam/RNG continuation, rather than testing serialization alone.
    def new_model():
        m=torch.nn.Module();m.vlm=torch.nn.Linear(3,3);m.head=torch.nn.Linear(3,1);m.vlm.requires_grad_(False);return m
    model=new_model();opt=torch.optim.AdamW(model.head.parameters(),lr=1e-5,betas=(.9,.95),weight_decay=.1,foreach=False)
    sched=torch.optim.lr_scheduler.ConstantLR(opt,factor=1.,total_iters=1)
    def one_step(m,o,s):
        x=torch.randn(2,3);target=torch.randn(2,1);o.zero_grad(set_to_none=True)
        loss=(m.head(x)-target).square().mean();loss.backward();o.step();s.step();return float(loss.detach())
    one_step(model,opt,sched)
    rp=ROOT/'tmp/exact_resume_cpu_fixture.pt';provenance={'fixture':True}
    training.save_resume(rp,model,opt,sched,dict(step=1),provenance)
    reference_loss=one_step(model,opt,sched);reference=training.active_state(model)
    other=new_model();op2=torch.optim.AdamW(other.head.parameters(),lr=1e-5,betas=(.9,.95),weight_decay=.1,foreach=False)
    sc2=torch.optim.lr_scheduler.ConstantLR(op2,factor=1.,total_iters=1)
    restored=training.restore_resume(rp,other,op2,sc2,provenance);assert restored['step']==1
    actual_loss=one_step(other,op2,sc2)
    assert reference_loss==actual_loss and all(torch.equal(reference[k],v) for k,v in training.active_state(other).items())
    assert not torch.cuda.is_initialized()
    return dict(**production,preserve_rng_restores_python_numpy_torch=True,
                shared_loader_preserves_source_fp32=True,source_vlm_remains_bf16=True,
                adam_rng_resume_exact_next_update=True,cuda_initialized=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--refresh',action='store_true');parser.add_argument('--smoke-only',action='store_true');args=parser.parse_args()
    if args.smoke_only:
        print(json.dumps(cpu_runtime_smoke(),indent=2));return
    target=ROOT/'prepared_checks.json'
    if target.exists() and not args.refresh:raise FileExistsError('Use train_ab --check-prepared or explicit pre-run --refresh')
    if list((ROOT/'arms').glob('*/run_manifest.json')):raise RuntimeError('Cannot rebuild preparation after any arm started')
    started=time.monotonic();cfg=training.configuration();selection=json.loads(common.SELECTION.read_text())
    assert json.loads((ROOT/'data_validation.json').read_text())['status']=='PASS'
    cache_validation=json.loads((ROOT/'cache_validation.json').read_text());assert cache_validation['status']=='PASS'
    assert file_sha256(common.CACHE)==cache_validation['eval_cache_sha256']
    assert file_sha256(common.SELECTION)==cache_validation['selection_sha256']
    assert file_sha256(common.STATS)==selection['stats_sha256']==cache_validation['stats_sha256']
    episodes={e['episode_id']:e for e in selection['episodes_train']};assert len(episodes)==128
    train_seeds={e['seed'] for e in episodes.values()}
    eval_windows=[w for group in selection['evaluation_sets'].values() for w in group]
    val_ids={w['episode_id'] for w in eval_windows if w['split']=='val'}
    val_seeds={w['seed'] for w in eval_windows if w['split']=='val'}
    heldout_ids={e['episode_id'] for e in selection['heldout_scenes']};heldout_seeds={e['seed'] for e in selection['heldout_scenes']}
    assert len(heldout_ids)==12 and not(set(episodes)&val_ids) and not(set(episodes)&heldout_ids) and not(val_ids&heldout_ids)
    assert not(train_seeds&val_seeds or train_seeds&heldout_seeds or val_seeds&heldout_seeds)
    for e in list(episodes.values())+selection['episodes_new_val']+selection['episodes_heldout']:
        assert file_sha256(e['annotation'])==e['annotation_sha256']
    # Complete cohort partition is checked against timestamps, never trace/frame one-to-one indexing.
    trajectories={};lookup_group={}
    for eid,e in episodes.items():
        t=json.loads(Path(e['annotation']).read_text());assert t['split']=='train' and t['episode_id']==eid
        trajectories[eid]=t;assigned=[]
        for group,frames in e['group_frames'].items():
            for f in frames:
                assigned.append(f);actual='reset' if f==0 else 'first1_4' if f<5 else common.phase_metadata(t,f)[0]
                assert actual==group,(eid,f,group,actual)
                lookup_group[(eid,f)]=group
        assert sorted(assigned)==list(range(t['num_frames']-29)) and len(assigned)==len(set(assigned))
    schedule_reports={};schedules={}
    for arm in ('A','B'):
        rows=training.load_schedule(arm);schedules[arm]=rows
        for row in rows:
            eid=row['episode_id'];assert eid in episodes and row.get('split','train')=='train'
            ep=episodes[eid];f=int(row['frame'])
            assert row['group']==lookup_group[(eid,f)] and row['seed']==ep['seed']
            assert Path(row['annotation']).resolve()==Path(ep['annotation']).resolve()
            assert row['window_id']==f'train/{eid}/frame{f:04d}'
        counts=Counter(r['group'] for r in rows)
        report=dict(rows=len(rows),unique_windows=len({r['window_id'] for r in rows}),group_counts=dict(counts),episode_count=len({r['episode_id'] for r in rows}))
        if arm=='A':assert report['unique_windows']==8000
        else:
            assert dict(counts)==cfg['B_quotas'];balance={}
            for group in cfg['B_quotas']:
                eligible=[eid for eid,e in episodes.items() if e['group_frames'][group]]
                c=Counter(r['episode_id'] for r in rows if r['group']==group);values=[c[eid] for eid in eligible]
                assert set(c)==set(eligible) and max(values)-min(values)<=1
                balance[group]=dict(eligible_episodes=len(eligible),min_presentations=min(values),max_presentations=max(values))
            report['episode_balance']=balance
        schedule_reports[arm]=report
    cache=torch.load(common.CACHE,map_location='cpu',weights_only=True,mmap=True)
    assert len(cache['samples'])==240
    counts=Counter(s['meta']['evaluation_set'] for s in cache['samples']);assert dict(counts)==cfg['evaluation_sets']
    seen=set()
    for sample in cache['samples']:
        m=sample['meta'];b=sample['batch'];assert m['window_id'] not in seen;seen.add(m['window_id'])
        if m['split']=='train':assert m['episode_id'] in episodes
        else:assert m['episode_id'] not in episodes
        assert m['episode_id'] not in heldout_ids
        assert tuple(b['action'].shape)==(1,30,32) and b['action'].dtype==torch.float32
        assert torch.equal(b['action_mask'][0],torch.from_numpy(common.action_mask()))
        assert 'prefix_length' not in b
        gt=sample['gt_action_physical'];norm=(gt-cache['mean'])/(cache['std']+common.EPS)
        assert torch.equal(b['action'][0],norm)
        zero=common.to_device(b,'cpu',for_generation=True);assert torch.count_nonzero(zero['action'])==0
        zero.pop('action');zero.pop('state');assert 'action' in b and 'state' in b
    # Full dataset fingerprint plus actual schedule Subset/collate parity on two windows per group.
    inputs=training.ScheduledInputs(schedules['B'],limit=16);checks=[]
    for group in cfg['B_quotas']:
        matches=[i for i,r in enumerate(schedules['B']) if r['group']==group][:2]
        for i in matches:
            row=schedules['B'][i];batch=inputs.get(i);gt=common.encode_window(trajectories[row['episode_id']],row['frame'])
            expected=(gt-inputs.dataset.mean)/(inputs.dataset.std+common.EPS)
            assert np.array_equal(batch['action'][0].numpy(),expected)
            checks.append(dict(group=group,window_id=row['window_id'],dataset_index=inputs.indices[i],action_exact=True))
    tests=cpu_runtime_smoke()
    artifact_paths=[common.SELECTION,common.CACHE,common.STATS,ROOT/'data_validation.json',ROOT/'cache_validation.json',ROOT/'schedule_meta.json',training.schedule_path('A'),training.schedule_path('B')]
    artifact_paths += [Path(e['annotation']) for e in episodes.values()]
    artifact_paths += [Path(w['annotation']) for w in eval_windows if w['split']=='val']
    artifact_paths += [Path(e['annotation']) for e in selection['heldout_scenes']]
    artifacts={str(p):file_sha256(p) for p in artifact_paths}
    report=dict(schema='orchard_ab_prepared_checks_v1',status='PASS',selection_sha256=file_sha256(common.SELECTION),
        base_sha256=file_sha256(common.BASE),artifact_sha256=artifacts,source_sha256=training.source_hashes(),
        schedules=schedule_reports,evaluation_sets=dict(counts),heldout_episodes=12,train_val_heldout_episode_and_seed_disjoint=True,
        full_training_dataset_windows=len(inputs.dataset),full_train_stats_fingerprint=inputs.dataset.stats['source_sha256'],
        full_dataset_then_schedule_subset=True,schedule_action_contract_checks=checks,tests=tests,
        zero_gt_generation_input=True,masked_inactive_dims=True,elapsed_s=time.monotonic()-started,cuda_initialized=torch.cuda.is_initialized())
    assert not report['cuda_initialized'];common.write_json(target,report,replace=args.refresh)
    print(json.dumps(dict(status='PASS',report=str(target),elapsed_s=report['elapsed_s'],schedules=schedule_reports,evaluation_sets=dict(counts),tests=tests),indent=2))

if __name__=='__main__':main()
