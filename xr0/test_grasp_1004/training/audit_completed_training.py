#!/usr/bin/env python3
"""Independent CPU-only endpoint audit; no training source or artifact mutations."""
import bootstrap
from collections import Counter
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import numpy as np
import torch
from checkpoint_io import file_sha256


def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def compare(a,b,path=''):
    out=[]
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a,b):out.append(path)
    elif isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b):return [path+' [keys]']
        for k in a:out.extend(compare(a[k],b[k],path+'.'+str(k)))
    elif isinstance(a,(list,tuple)):
        if type(a)!=type(b) or len(a)!=len(b):return [path+' [sequence]']
        for i,(x,y) in enumerate(zip(a,b)):out.extend(compare(x,y,path+f'[{i}]'))
    elif type(a)!=type(b) or a!=b:out.append(path)
    return out


def metrics(rows):
    a=np.array([r['gradient_norm_before_clip'] for r in rows],dtype=float)
    return dict(updates=len(rows),gradient_norm_mean=float(a.mean()),gradient_norm_p50=float(np.median(a)),
        gradient_norm_p95=float(np.quantile(a,.95)),clipped_updates=int((a>1.).sum()),clipped_fraction=float((a>1.).mean()))


def main():
    torch.set_num_threads(1)
    root=bootstrap.ROOT;ab=bootstrap.AB;l=root/'arms/L';b=ab/'arms/B'
    ls,bs=read(l/'training_summary.json'),read(b/'training_summary.json')
    assert ls['status']==bs['status']=='COMPLETE' and ls['completed_updates']==bs['completed_updates']==8000
    ll,bl=lines(l/'training_steps.jsonl'),lines(b/'training_steps.jsonl');schedule=lines(ab/'schedules/B.jsonl')
    assert len(ll)==len(bl)==len(schedule)==8000
    for i,(lr,br,s) in enumerate(zip(ll,bl,schedule),1):
        assert lr['step']==br['step']==s['step']==i
        for k in ('window_id','episode_id','frame','group'):assert lr[k]==br[k]==s[k],(i,k)
        assert lr['split']==br['split']=='train'
        assert lr['lr']==br['lr']==1e-5
        assert lr['gradients_finite'] and br['gradients_finite']
        assert all(math.isfinite(r[k]) for r in (lr,br) for k in ('flow_loss','flow_loss_mse','gradient_norm_before_clip'))
        assert lr['gradient_clipped']==(lr['gradient_norm_before_clip']>1.)
    wr=lines(bootstrap.EXPERIMENT/'data/L_schedule_weights.jsonl')
    assert all(r['contact_targets']==sum(w['contact_flags']) for r,w in zip(ll,wr))
    for k in ('per_window_counts','per_episode_counts','group_counts'):assert ls[k]==bs[k]
    assert ls['val_optimizer_updates']==bs['val_optimizer_updates']==0
    assert ls['train_episodes_seen']==bs['train_episodes_seen']==128
    assert ls['unique_train_windows_seen']==bs['unique_train_windows_seen']==3931
    assert ls['trainable_master_dtype']==bs['trainable_master_dtype']=='float32'
    frozen={s[k] for s in (ls,bs) for k in ('frozen_vlm_parameter_sha256_before','frozen_vlm_parameter_sha256_after')}
    assert len(frozen)==1 and ls['frozen_vlm_unchanged'] and bs['frozen_vlm_unchanged']
    lm,bm=read(l/'run_manifest.json'),read(b/'run_manifest.json')
    assert lm['initial_optimizer_state_entries']==bm['initial_optimizer_state_entries']==0
    assert lm['configuration']['optimizer']==bm['configuration']['optimizer']
    for k in ('base_sha256','selection_sha256','schedule_sha256','stats_sha256','eval_cache_sha256'):assert lm['provenance'][k]==bm['provenance'][k]
    for manifest in (lm,bm):assert manifest['initial_load_report']['non_vlm_source_fp32_tensor_exact_equal'] and manifest['initial_load_report']['source_fp32_parameter_tensors']==219
    prepared=read(root/'prepared_checks.json')
    for path,digest in {**prepared['artifact_sha256'],**prepared['source_sha256']}.items():assert file_sha256(path)==digest,path
    baseline=read(l/'baseline_verification.json');startup=read(l/'first_100_verification.json')
    assert baseline['status']==startup['status']=='PASS' and baseline['arrays_exact']==4320 and startup['updates']==100
    rl=torch.load(l/'checkpoints/latest_resume.pt',map_location='cpu',weights_only=True,mmap=True)
    rb=torch.load(b/'checkpoints/latest_resume.pt',map_location='cpu',weights_only=True,mmap=True)
    assert rl['state']['step']==rb['state']['step']==8000
    rng={k:dict(exact_equal=not compare(rl['rng'][k],rb['rng'][k],k),mismatch_paths=compare(rl['rng'][k],rb['rng'][k],k)) for k in ('python','numpy','torch_cpu','torch_cuda')}
    optimizer_groups=not compare(rl['optimizer']['param_groups'],rb['optimizer']['param_groups'],'optimizer.param_groups')
    assert optimizer_groups
    assert len(rl['optimizer']['state'])==len(rb['optimizer']['state'])==219
    assert all(float(s['step'])==8000 for payload in (rl,rb) for s in payload['optimizer']['state'].values())
    target=l/'checkpoints/step_8000_trainable.pt';assert file_sha256(target)==ls['checkpoint_records']['8000']['checkpoint_sha256']
    overlay=torch.load(target,map_location='cpu',weights_only=True,mmap=True);state=overlay['trainable_state_dict']
    assert len(state)==219 and all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in state.values())
    assert not torch.cuda.is_initialized()
    report=dict(status='PASS' if all(x['exact_equal'] for x in rng.values()) else 'PASS_WITH_RNG_DIFFERENCE',created_utc=datetime.now(timezone.utc).isoformat(),
        updates=8000,exact_ordered_B_schedule=True,all_8000_losses_and_gradients_finite=True,
        train_episodes=128,unique_windows=3931,per_window_episode_and_group_counts_equal_B=True,val_optimizer_updates=0,
        same_initial_source_fp32_219_tensors=True,same_fresh_optimizer=True,optimizer_groups_equal_B=True,
        final_optimizer_state_entries=219,all_optimizer_steps=8000,frozen_vlm_unchanged_and_equal_B=True,
        frozen_vlm_parameter_sha256=next(iter(frozen)),training_sources_and_data_unchanged=True,
        baseline_240_windows_4320_arrays_exact=True,first100_finite=True,final_overlay_fp32_tensors=219,
        final_overlay_checkpoint_sha256=file_sha256(target),
        rng_state_comparison_against_B8000=rng,rng_exact_all=all(x['exact_equal'] for x in rng.values()),
        rng_interpretation='Exact states mean weighting and added auxiliary evaluations did not alter Python/NumPy/CPU/CUDA random-number consumption; this does not require learned weights or gradients to match.',
        gradient_clipping=dict(L=metrics(ll),B=metrics(bl),first100_L=metrics(ll[:100]),first100_B=metrics(bl[:100]),
            by_group={g:dict(L=metrics([r for r in ll if r['group']==g]),B=metrics([r for r in bl if r['group']==g])) for g in sorted(ls['group_counts'])}),
        loss_comparison_note='L and B training losses use different target weights, so raw loss means are not direct skill comparisons.',
        no_source_or_checkpoint_mutations=True,cuda_initialized=False,audit_source_sha256=file_sha256(__file__))
    path=root/'post_training_audit.json'
    with path.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=report['status'],report=str(path),rng_exact_all=report['rng_exact_all'],gradient_clipping=report['gradient_clipping'])))

if __name__=='__main__':main()
