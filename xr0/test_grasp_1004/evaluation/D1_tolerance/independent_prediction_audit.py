#!/usr/bin/env python3
"""Independent CPU-only D1 initial-observation and first decoded-chunk audit.

--check-interface uses D0 records only; --run requires all eight D1 completions.
Never imports simulator/model modules, never changes a source/protocol/checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
D0=ROOT.parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def first(path):
    with Path(path).open() as f:return json.loads(next(line for line in f if line.strip()))

def mismatches(a,b,path=''):
    if type(a)!=type(b):return [path+' [type]']
    if isinstance(a,dict):
        if set(a)!=set(b):return [path+' [keys]']
        return [m for k in a for m in mismatches(a[k],b[k],path+'.'+k)]
    if isinstance(a,list):
        if len(a)!=len(b):return [path+' [length]']
        return [m for i,(x,y) in enumerate(zip(a,b)) for m in mismatches(x,y,path+f'[{i}]')]
    return [] if a==b else [path]


def validate_completion(folder,protocol_hash,checkpoint_hash):
    d=read(folder/'completed.json');assert d['status']=='COMPLETE' and d['protocol_sha256']==protocol_hash
    assert d['checkpoint_sha256']==checkpoint_hash
    for name,digest in d['files_sha256'].items():assert sha(folder/name)==digest,(folder,name)
    return d


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--check-interface',action='store_true');g.add_argument('--run',action='store_true');args=p.parse_args()
    protocol=read(ROOT/'protocol.json');d0=read(D0/'protocol.json')
    assert sha(D0/'protocol.json')==protocol['d0_protocol_sha256']
    fields=('chunk_id','at_control_step','anchor','start_k','stop_k','target_positions','target_rotations','target_widths','terminal_hold','processed_targets_per_replan')
    rows=[]
    for stage in protocol['queue_order']:
        for job in protocol['jobs'][stage]:
            provider='GT_targets30' if job['is_gt'] else 'B8000_targets30_rng42'
            old=D0/'rollouts'/f'{provider}_reach-conditioned_{job["scene"]["seed"]}'
            validate_completion(old,sha(D0/'protocol.json'),None if job['is_gt'] else d0['checkpoint']['sha256'])
            oi=read(old/'initial.json');oc=first(old/'chunks.jsonl')
            assert len(oc['target_positions'])==len(oc['target_rotations'])==len(oc['target_widths'])==30
            assert oc['at_control_step']==oc['chunk_id']==oc['start_k']==0 and oc['stop_k']==30
            if job['is_gt']:
                assert oc['anchor']['recorded_frame']==0
            if not job['is_gt']:
                assert oc['anchor']==oi['obs']
                assert oi['actual_inference_seed']==oc['actual_inference_seed']==42
                seeds=read(old/'seed_delivery.json');assert seeds['declared']==seeds['calls'][0]['actual_inference_seed']==42
            if args.check_interface:
                assert not mismatches(oi['obs'],oi['obs'])
                changed=json.loads(json.dumps(oc));changed['target_positions'][0][0]+=.001
                assert mismatches(oc['target_positions'],changed['target_positions'])==['[0][0]']
                rows.append(dict(stage=stage,seed=job['scene']['seed'],baseline=str(old),baseline_verified=True));continue
            current=ROOT/'rollouts'/job['job_id']
            validate_completion(current,sha(ROOT/'protocol.json'),None if job['is_gt'] else protocol['checkpoint']['sha256'])
            ci=read(current/'initial.json');cc=first(current/'chunks.jsonl')
            obs_diff=mismatches(oi['obs'],ci['obs'],'obs')
            chunk_diff={key:mismatches(oc[key],cc[key],key) for key in fields}
            if not job['is_gt']:
                assert ci['actual_inference_seed']==cc['actual_inference_seed']==42
                delivery=read(current/'seed_delivery.json');assert delivery['declared']==delivery['calls'][0]['actual_inference_seed']==42
            trace=[json.loads(line) for line in (current/'steps.jsonl').read_text().splitlines() if line.strip()]
            candidates=[r for r in trace if .005<r['info']['ik_error']['position_m']<=.01 and r['info']['ik_error']['rotation_rad']<=.08]
            held=0
            for r in candidates:
                j=r['joint_command_delta'];reconstructed=[q-p for p,q in zip(j['pre_joint_command'],j['post_joint_command'])]
                assert all(abs(a-b)<=1e-12 for a,b in zip(reconstructed,j['delta']))
                assert r['info']['ik_failed'] is True
                held+=all(abs(x)<=1e-12 for x in reconstructed[:7])
            met=read(current/'metrics.json');assert met['position_threshold_only_rejection_count']==len(candidates)
            assert met['position_threshold_candidates_with_held_arm_command']==held
            rows.append(dict(stage=stage,seed=job['scene']['seed'],initial_observation_exact=not obs_diff,
                initial_observation_mismatch_paths=obs_diff,first_decoded_chunk_exact=not any(chunk_diff.values()),
                first_decoded_chunk_mismatch_paths=chunk_diff,source_model_seed_unchanged=True,
                extra_position_rejection_candidates=len(candidates),all_candidates_recorded_ik_failed=True,
                candidate_arm_command_held=held,candidate_counter_reconstruction_exact=True))
    result=dict(status='INTERFACE_PASS' if args.check_interface else 'PASS' if all(r['initial_observation_exact'] and r['first_decoded_chunk_exact'] for r in rows) else 'PARITY_DIFFERENCE',
        cpu_only=True,gpu_used=False,protocol_sha256=sha(ROOT/'protocol.json'),audit_source_sha256=sha(__file__),rows=rows,
        executed_D1_comparison=not args.check_interface,
        interpretation='Initial observation includes exact RGB hashes and state values. First chunk compares stored decoded world positions/rotations/absolute widths for all30 targets, not unrecorded normalized model tensors. Command held means target unchanged, not measured arm motion stopped. No tolerance is used for initial/model prediction comparisons.')
    target=ROOT/('independent_prediction_interface.json' if args.check_interface else 'independent_prediction_audit.json')
    with target.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],rows=len(rows),output=str(target))))

if __name__=='__main__':main()
