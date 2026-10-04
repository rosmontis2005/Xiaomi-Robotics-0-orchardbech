#!/usr/bin/env python3
"""Freeze CPU provenance, event alignment and unchanged training inputs for L."""
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
import train_l as training
import cpu_loss_checks
from checkpoint_io import file_sha256
ROOT=bootstrap.ROOT


def cpu_resume_and_precision():
    from mibot.utils.orchard_checkpoint import load_weights
    m=torch.nn.Module();m.vlm=torch.nn.Linear(3,3).bfloat16();m.head=torch.nn.Linear(3,1).bfloat16()
    source={k:v.clone() for k,v in m.state_dict().items()}
    source['head.weight']=torch.tensor([[.10001,.20003,.30007]],dtype=torch.float32)
    source['head.bias']=torch.tensor([.01234567],dtype=torch.float32)
    p=ROOT/'tmp/source_precision_fixture.pt';torch.save({'state_dict':source},p);load_weights(m,p)
    assert m.head.weight.dtype==torch.float32 and torch.equal(m.head.weight,source['head.weight'])
    assert m.vlm.weight.dtype==torch.bfloat16
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    state=training.capture_rng();expected=(random.random(),np.random.rand(),torch.randn(3))
    training.restore_rng(state)
    with common.preserve_rng():random.seed(99);np.random.seed(99);torch.manual_seed(99);torch.randn(3)
    actual=(random.random(),np.random.rand(),torch.randn(3))
    assert expected[:2]==actual[:2] and torch.equal(expected[2],actual[2])
    def create():
        n=torch.nn.Module();n.vlm=torch.nn.Linear(3,3);n.head=torch.nn.Linear(3,1);n.vlm.requires_grad_(False)
        o=torch.optim.AdamW(n.head.parameters(),lr=1e-5,betas=(.9,.95),weight_decay=.1,foreach=False)
        s=torch.optim.lr_scheduler.ConstantLR(o,factor=1.,total_iters=1)
        return n,o,s
    def update(n,o,s):
        x=torch.randn(2,3);y=torch.randn(2,1);o.zero_grad(set_to_none=True);l=(n.head(x)-y).square().mean();l.backward();o.step();s.step();return l.item()
    n,o,s=create();update(n,o,s);p=ROOT/'tmp/resume_fixture.pt';training.save_resume(p,n,o,s,dict(step=1),dict(test=True))
    loss=update(n,o,s);weights=training.active_state(n)
    b,bo,bs=create();restored=training.restore_resume(p,b,bo,bs,dict(test=True));assert restored['step']==1
    assert update(b,bo,bs)==loss and all(torch.equal(v,weights[k]) for k,v in training.active_state(b).items())
    return dict(precision_preserved=True,rng_restored=True,adam_resume_next_update_exact=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--refresh',action='store_true');args=p.parse_args()
    target=ROOT/'prepared_checks.json'
    if list((ROOT/'arms').glob('*/run_manifest.json')):raise RuntimeError('Cannot rebuild preparation after L has started')
    if target.exists() and not args.refresh:raise FileExistsError('Use --refresh only before launch')
    started=time.monotonic();torch.set_num_threads(1)
    old=json.loads((bootstrap.AB/'prepared_checks.json').read_text());assert old['status']=='PASS'
    for path,sha in {**old['artifact_sha256'],**old['source_sha256']}.items():assert file_sha256(path)==sha,path
    training.ensure_precision_ready()
    cfg=training.configuration();oldcfg=json.loads((bootstrap.AB/'config_ab.json').read_text())
    for key in ('updates_per_arm','evaluation_steps','prediction_seeds','train_seed','batch_size','gradient_accumulation','optimizer','gradient_clip_norm','freeze_vlm','training_repeat','async_train','enable_freq','generation_euler_steps','checkpoint_steps','resume_every_updates','cpu_input_lru_windows','evaluation_sets'):
        assert cfg[key]==oldcfg[key],key
    rows=training.load_schedule('L');events=json.loads(training.EVENTS.read_text())
    weight_rows=[json.loads(line) for line in training.WEIGHTS.read_text().splitlines() if line.strip()]
    assert len(weight_rows)==8000 and events['schema']=='orchard_contact_event_index_v1'
    selected=json.loads(common.SELECTION.read_text());train_ids={r['episode_id'] for r in selected['episodes_train']}
    assert len(train_ids)==128 and {r['episode_id'] for r in rows}==train_ids
    artifact_paths=[bootstrap.EXPERIMENT/'data/independent_data_validation.json',bootstrap.EXPERIMENT/'data/aux_cache_validation.json',training.EVENTS,training.WEIGHTS,training.AUX_CACHE,common.CACHE,common.SELECTION,common.STATS,training.schedule_path('L'),bootstrap.AB/'prepared_checks.json',bootstrap.AB/'precision_fix.json']
    contact=0;normalized_mass=0.
    for row,wr in zip(rows,weight_rows):
        for key in ('step','episode_id','split','frame','window_id','annotation_sha256'):assert wr[key]==row[key],(row['step'],key)
        assert row['split']=='train'
        ep=events['episodes'][f"train/{row['episode_id']}"]
        assert ep['annotation_sha256']==row['annotation_sha256']==file_sha256(row['annotation'])
        idx=np.minimum(np.arange(row['frame']+1,row['frame']+31),ep['num_frames']-1)
        assert wr['target_indices']==idx.tolist()
        ts=np.array(ep['timestamps_s'])[idx]
        flags=(ts>=ep['interval_start_s']-1e-9)&(ts<=ep['interval_end_s']+1e-9)
        assert flags.tolist()==wr['contact_flags'],row['window_id']
        assert np.array_equal(np.array(wr['raw_target_weights']),1.+flags)
        contact+=int(flags.sum());normalized_mass+=float((2.*flags).sum()/(30+flags.sum()))
    artifact_paths += [Path(e['annotation']) for e in events['episodes'].values()]
    cache=torch.load(common.CACHE,map_location='cpu',weights_only=True,mmap=True)
    aux=torch.load(training.AUX_CACHE,map_location='cpu',weights_only=True,mmap=True)
    assert len(cache['samples'])==240 and len(aux['samples'])==144
    assert torch.equal(cache['mean'],aux['mean']) and torch.equal(cache['std'],aux['std'])
    for panel in (cache,aux):
        windows=set()
        for sample in panel['samples']:
            m=sample['meta'];b=sample['batch'];assert m['window_id'] not in windows;windows.add(m['window_id'])
            assert tuple(b['action'].shape)==(1,30,32) and b['action'].dtype==torch.float32
            assert torch.equal(b['action_mask'][0],torch.from_numpy(common.action_mask()))
            assert set(b).isdisjoint({'event_loss_weight','contact_flags','event_time_s','prefix_length'})
            assert (m['episode_id'] in train_ids)==(m['split']=='train')
            before={k:v.clone() for k,v in b.items()}
            gen=common.to_device(b,'cpu',for_generation=True);assert torch.count_nonzero(gen['action'])==0
            gen.pop('action');gen.pop('state');assert all(torch.equal(v,b[k]) for k,v in before.items())
    inputs=training.ScheduledInputs(rows,limit=16);checks=[]
    for group in cfg['B_quotas']:
        for i in [i for i,r in enumerate(rows) if r['group']==group][:2]:
            row=rows[i];batch=inputs.get(i);traj=json.loads(Path(row['annotation']).read_text())
            gt=common.encode_window(traj,row['frame']);expected=(gt-inputs.dataset.mean)/(inputs.dataset.std+common.EPS)
            assert np.array_equal(batch['action'][0].numpy(),expected)
            checks.append(dict(group=group,window_id=row['window_id'],action_exact=True))
    tests=dict(loss=cpu_loss_checks.run(),runtime=cpu_resume_and_precision(),overlay=common.cpu_objective_and_overlay_test())
    assert not torch.cuda.is_initialized()
    report=dict(schema='orchard_contact_loss_prepared_v1',status='PASS',base_sha256=file_sha256(common.BASE),
        artifact_sha256={str(p):file_sha256(p) for p in artifact_paths},source_sha256=training.source_hashes(),
        old_prepared_sha256=file_sha256(bootstrap.AB/'prepared_checks.json'),
        original_config_controls_unchanged=True,same_B_schedule_rows=8000,training_episodes=128,
        full_train_stats_fingerprint=inputs.dataset.stats['source_sha256'],full_training_dataset_windows=len(inputs.dataset),
        contact_targets=contact,total_targets=240000,raw_contact_fraction=contact/240000,
        normalized_effective_contact_fraction=normalized_mass/8000,
        old_eval_windows=240,aux_eval_windows=144,aux_does_not_select_checkpoint=True,
        sample_contract_checks=checks,tests=tests,cuda_initialized=False,elapsed_s=time.monotonic()-started)
    common.write_json(target,report,replace=args.refresh)
    print(json.dumps(dict(status='PASS',report=str(target),elapsed_s=report['elapsed_s'],tests=tests)))

if __name__=='__main__':main()
