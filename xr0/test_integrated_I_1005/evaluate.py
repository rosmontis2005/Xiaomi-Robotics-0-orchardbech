"""Unchanged reach-conditioned full30 production evaluator, unified S/J metrics."""
import bootstrap
import argparse,json,time,os,sys,traceback,fcntl
from pathlib import Path
from collections import Counter
import numpy as np
from recovery_common import *
from checkpoint_io import load_trainable_overlay

D0=XR0/'test_grasp_1004/evaluation';sys.path.append(str(D0));bootstrap.AB=AB

def lines(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

def evaluate(endpoint,checkpoint):
    out=ROOT/'round1'/endpoint/'evaluation';out.mkdir(parents=True,exist_ok=True)
    def status(phase,**extra):write(out/'status.json',dict(status=phase,pid=os.getpid(),time=time.time(),**extra))
    lock=(XR0/'test_recovery_R_1004/.gpu_training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        import torch
        torch.set_num_threads(1);protection_check();d0=json.loads((D0/'protocol.json').read_text());selection=AB/'selection.json';assert sha(STATS)==d0['stats_sha256'] and sha(selection)==d0['selection_sha256']
        jobs=[]
        for stage in ['B30','regression_B30']:
            for prior in d0['jobs'][stage]:
                j=dict(prior);j.update(stage='integrated30' if stage=='B30' else 'regression_integrated30',provider=prior['provider'].replace('B8000',endpoint),job_id=prior['job_id'].replace('B8000',endpoint),baseline_job_id=prior['job_id']);jobs.append(j)
        protocol=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),jobs=jobs,reach=d0['reach'],strict=d0['strict'],selection_sha256=sha(selection),stats_sha256=sha(STATS),no_fresh24=True,endpoint=endpoint,code_sha256=sha(Path(__file__)))
        if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
        else:write(out/'protocol.json',protocol)
        status('MODEL_LOADING')
        from mibot.server.orchard_policy import OrchardPolicy
        policy=OrchardPolicy(str(BASE),str(PROCESSOR),str(STATS));report=load_trainable_overlay(policy.model,checkpoint,base_checkpoint=BASE);assert report['step']==(16000 if endpoint=='J16' else 8000)
        policy.model.num_steps=5;write(out/'load_manifest.json',report)
        loop=module(D0/'reach_loop.py','i_reach');loop.ROOT=out;loop.load_runtime()
        helpers=module(AB/'closed_loop_1004/evaluate_ab.py','i_helpers');helpers.ROOT=out
        analyze=module(XR0/'diagnosis_1003/analyze_rollouts.py','i_analyzer');passive=module(AB/'closed_loop_1004/passive_metrics.py','i_passive')
        summarize=module(XR0/'test_history_H_1005/summarize_cases.py','i_cases')
        active={};base_line=loop.line;base_snapshot=loop.snapshot_grasp;results=[];cases=[]
        def line(stream,data):
            data.update(inference_seed=active['inference_seed'],job_id=active['job_id']);base_line(stream,data)
            if 'control_step' in data and data['control_step']%150==0:status('CLOSED_LOOP',job=active['job_id'],control_step=data['control_step'],completed=len(results))
        def snapshot(env):
            s=base_snapshot(env);idx=active['planned_apple_id'];body=int(env.tm.apple_bodies[idx]);s['planned_apple']=passive.planned_apple_observation(s,idx,body,env.sim.body_q_np()[body,:3])
            from treesim import robot
            cp,cr=env._chassis_pose();bucket=cp+cr.apply([robot._BUCKET_CENTER_X,0.,robot._CHASSIS_Z+robot._CHASSIS[2]+robot._BUCKET_WALL_H]);apple=env.sim.body_q_np()[body,:3]
            s['placement_geometry']=dict(diagnostic_only=True,bucket_top_center_world=bucket.tolist(),planned_fruit_bucket_distance_m=float(np.linalg.norm(apple-bucket)),planned_fruit_base_local=cr.inv().apply(apple-cp).tolist(),base_position=cp.tolist(),base_quaternion=cr.as_quat().tolist())
            ids=active.setdefault('observed_fruit_ids',set());ids.add(idx)
            if s['held_apple_id'] is not None:ids.add(int(s['held_apple_id']))
            poses=env.sim.body_q_np();s['fruit_bucket_geometry']={str(i):dict(world_position=poses[int(env.tm.apple_bodies[i]),:3].tolist(),distance_m=float(np.linalg.norm(poses[int(env.tm.apple_bodies[i]),:3]-bucket)),base_local=cr.inv().apply(poses[int(env.tm.apple_bodies[i]),:3]-cp).tolist()) for i in sorted(ids)}
            return s
        loop.line=line;loop.snapshot_grasp=snapshot
        for j in jobs:
            folder=out/'rollouts'/j['job_id'];active.clear();active.update(j);traj=json.loads(Path(j['scene']['annotation']).read_text());active['planned_apple_id']=int(traj['orchardbench']['fixed_base_expert']['selected_apple_debug_index'])
            if (folder/'completed.json').exists():r=json.loads((folder/'summary.json').read_text())
            else:
                assert not folder.exists(),('Partial rollout retained; inspect before retry',folder)
                bound=helpers.BoundSeedPolicy(policy,j['inference_seed']);r=loop.episode(bound,j['provider'],j['scene'],str(checkpoint),True,sha(selection),processed_targets=30,is_gt=False);assert r['termination']!='error'
                steps=lines(folder/'steps.jsonl');chunks=lines(folder/'chunks.jsonl');metrics=analyze.analyze_episode(r,steps,chunks,folder);metrics.update(helpers.held_runs(steps))
                gtref=AB/'closed_loop_1004/rollouts'/f'gt_reach-conditioned_{j["scene"]["seed"]}'/'chunks.jsonl';metrics.update(passive.diagnostics(steps,chunks,lines(gtref)));metrics.update(strict_placement=r['strict_placement'],strict_success=r['strict_success'],legacy_success_ever=r['legacy_success_ever'])
                assert json.loads((folder/'initial.json').read_text())['obs']==json.loads((D0/'rollouts'/j['baseline_job_id']/'initial.json').read_text())['obs']
                assert r['scene_check_pass'] and metrics['observer_consistency']['mismatch_fields_total']==0 and metrics['summary_consistency']['mismatch_count']==0
                write(folder/'metrics.json',metrics);write(folder/'seed_delivery.json',dict(calls=bound.calls));write(folder/'completed.json',dict(status='COMPLETE',checkpoint_sha256=sha(checkpoint),baseline_reset_exact=True))
            case=summarize.summarize_case(folder,'historical_seed42_43' if j['stage'].startswith('regression') else 'dev8',j['inference_seed'])
            # Preserve the historical observer unchanged. Primary reporting
            # additionally requires the user's reasonable release for the SAME
            # strict-success chain; retain raw observer result for provenance.
            case['strict_original_observer']=case['strict']
            successes=r['strict_placement']['strict_success_events']
            case['strict']=case['strict'] and any(e['valid_chain'] and e['reasonable_region'] is True and any(z['apple_id']==e['apple_id'] and z['release_step']==e['step'] for z in successes) for e in case['release_events'])
            case['stable_bucket']=case['max_stable_bucket_steps']>=60;case['premature_release']=any(e['valid_chain'] and e['reasonable_region'] is False for e in case['release_events'])
            cases.append(case);results.append(r);write(out/'cases.json',cases);print('CASE',json.dumps({k:case[k] for k in ['seed','held15','same_fruit_detach','bucket_neighborhood','reasonable_release','stable_bucket','strict']}),flush=True)
        groups=summarize.aggregate(cases)
        for cohort,group in groups.items():
            subset=[c for c in cases if c['cohort']==cohort or cohort=='dev8_train2' and c['cohort']=='dev8' and c['split']=='train' or cohort=='dev8_val6' and c['cohort']=='dev8' and c['split']=='val']
            group['strict_original_observer']=sum(c['strict_original_observer'] for c in subset);group['stable_bucket']=sum(c['stable_bucket'] for c in subset);group['premature_release']=sum(c['premature_release'] for c in subset)
        write(out/'results.json',results);write(out/'aggregate.json',groups);status('COMPLETE',completed=len(cases),strict=groups['dev8']['strict'])
        import csv
        keys=['seed','inference_seed','cohort','held15','same_fruit_detach','bucket_neighborhood','reasonable_release','stable_bucket','strict','long_held_stall','premature_release','physical_anomaly','min_held_fruit_bucket_distance_m','failure']
        with (out/'cases.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(cases)
    except BaseException as e:status('FAILED',error=str(e),traceback=traceback.format_exc());raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--endpoint',choices=['S8','J8','J16'],required=True);p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args();evaluate(a.endpoint,a.checkpoint)
