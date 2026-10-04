"""Independent post-completion CPU summary of the fixed 20GT +64 A/B primary runs.
No simulation, policy import, model selection, or modification of raw results.
"""
import argparse, csv, hashlib, json, math, statistics, sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
LOOP=ROOT/'closed_loop_1004'
BINARY=['grasp','held15','grasp_by_step300','held15_qualified_by_step300','any_detach','grasp_and_any_detach','success']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(path.read_text())
def write(path,obj):
    assert path.resolve().is_relative_to(HERE)
    with path.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')

def stats(values):
    v=[float(x) for x in values if x is not None]
    assert all(math.isfinite(x) for x in v)
    return None if not v else dict(count=len(v),mean=statistics.mean(v),median=statistics.median(v),min=min(v),max=max(v))

def pair(a,b):
    assert len(a)==len(b)
    return dict(pairs=len(a),A_only=sum(bool(x) and not bool(y) for x,y in zip(a,b)),B_only=sum(bool(y) and not bool(x) for x,y in zip(a,b)),both=sum(bool(x) and bool(y) for x,y in zip(a,b)),neither=sum(not bool(x) and not bool(y) for x,y in zip(a,b)),B_minus_A_success_count=sum(map(bool,b))-sum(map(bool,a)))

def distance(m,name):
    value=m[name]
    return value['apple']['tcp_distance_m'] if value and value.get('apple') else None

def independently_reconstruct(steps,summary,metrics,job):
    assert len(steps)==summary['control_steps'] and len(steps)>0
    assert [r['control_step'] for r in steps]==list(range(1,len(steps)+1))
    assert len(steps)<=900
    first_grasp=None;first_success=None;first_detach=None
    last_id=None;run_start=None;run_length=0;maximum=0;sustained=[];releases=[];ik=0;clipped=0;timeouts=0
    for r in steps:
        step=r['control_step'];info=r['info'];held=r['grasp_after']['held_apple_id']
        assert r['job_id']==job['job_id'] and r['inference_seed']==job['inference_seed']
        assert all(r['observer_consistency'].values())
        if first_grasp is None and (info['grasp_assist_triggered'] or held is not None):first_grasp=step
        if first_success is None and info['success']:first_success=r
        if first_detach is None and info['apple_detached_count']>0:first_detach=step
        if held is None:run_length=0;run_start=None
        elif held!=last_id:run_length=1;run_start=step
        else:run_length+=1
        if run_length==15:sustained.append((run_start,step,held))
        maximum=max(maximum,run_length);last_id=held
        if r['grasp_before']['held_apple_id'] is not None and held is None:releases.append(step)
        ik+=bool(info['ik_failed']);clipped+=bool(info['action_clipped'])
        reached=r['position_error_m']<=.01 and r['rotation_error_rad']<=.08
        expired=not reached and r['dwell_steps']>=30
        reason='reached' if reached else 'maximum_dwell' if expired else 'hold'
        assert r['reached']==reached and r['advance']==(reached or expired) and r['advance_reason']==reason
        timeouts+=expired
    assert summary['ever_grasped']==(first_grasp is not None)==metrics['actual_grasp_event']
    assert metrics['first_grasp_control_step']==first_grasp and metrics['first_detach_control_step']==first_detach
    assert metrics['same_fruit_held_at_least_15_steps']==bool(sustained)
    assert metrics['max_same_fruit_held_steps']==maximum
    assert metrics['first_held15_qualification_step']==(sustained[0][1] if sustained else None)
    assert metrics['first_sustained_grasp_start_step']==(sustained[0][0] if sustained else None)
    assert metrics['held15_qualified_by_step300']==any(r[1]<=300 for r in sustained)
    assert metrics['sustained_grasp_started_by_step300']==any(r[0]<=300 for r in sustained)
    assert metrics['grasp_by_step300']==(first_grasp is not None and first_grasp<=300)
    assert metrics['any_detach']==(first_detach is not None)
    assert metrics['grasp_and_any_detach']==(first_grasp is not None and first_detach is not None)
    assert summary['success']==(first_success is not None)
    assert metrics['observed_held_to_none_release_count']==len(releases)
    if first_success:
        assert metrics['held_apple_id_at_first_success']==first_success['grasp_after']['held_apple_id']
        assert metrics['release_observed_strictly_before_success']==any(x<first_success['control_step'] for x in releases)
        assert metrics['release_observed_by_success_boundary']==any(x<=first_success['control_step'] for x in releases)
    else:
        assert metrics['held_apple_id_at_first_success'] is None
        assert metrics['release_observed_strictly_before_success'] is None and metrics['release_observed_by_success_boundary'] is None
    assert ik==metrics['ik_failed_steps'] and clipped==metrics['clipping']['union_steps']
    assert timeouts==metrics['advancement']['dwell_timeouts']
    return dict(raw_steps=len(steps),grasp_held15_300_success_release_recomputed=True,reach_position_rotation_max_dwell_recomputed=True,IK_clipping_timeout_counts_recomputed=True)

def collect(job,protocol_hash,endpoints):
    directory=LOOP/'rollouts'/job['job_id'];done=read(directory/'completed.json')
    expected_sha=None if job['stage']=='GT' else endpoints[job['arm']]['sha256']
    assert done['status']=='COMPLETE' and done['job_id']==job['job_id']
    assert done['protocol_sha256']==protocol_hash and done['checkpoint_sha256']==expected_sha
    assert done['inference_seed']==job['inference_seed'] and done['annotation_sha256']==job['scene']['annotation_sha256']
    assert sha(job['scene']['annotation'])==job['scene']['annotation_sha256']
    for name,digest in done['files_sha256'].items():assert sha(directory/name)==digest,('artifact SHA mismatch',directory/name)
    summary=read(directory/'summary.json');metrics=read(directory/'metrics.json');reset=read(directory/'scene_check.json');seed=read(directory/'seed_delivery.json')
    assert summary['termination']!='error' and summary['scene_check_pass'] and reset['state_and_identity_pass']
    assert metrics['observer_consistency']['mismatch_fields_total']==0 and metrics['summary_consistency']['mismatch_count']==0
    assert summary['seed']==job['scene']['seed'] and summary['episode_id']==job['scene']['episode_id']
    assert summary['inference_seed']==metrics['inference_seed']==job['inference_seed']
    if job['stage']!='GT':
        assert summary['checkpoint_step']==metrics['checkpoint_step']==8000
        assert summary['comparison_role']==metrics['comparison_role']=='primary_endpoint'
        assert reset['current_gt_pairing']['exact_match'] and reset['current_gt_pairing']['gt_runtime_valid']
        assert seed['all_calls_match'] and len(seed['calls'])==summary['replans']
        assert all(c['actual_inference_seed']==job['inference_seed'] for c in seed['calls'])
    assert metrics['actual_grasp_event']==summary['ever_grasped']==done['grasp']
    assert metrics['same_fruit_held_at_least_15_steps']==done['held15']
    assert summary['success']==done['success']
    steps=[json.loads(x) for x in (directory/'steps.jsonl').read_text().splitlines()]
    raw_check=independently_reconstruct(steps,summary,metrics,job)
    row=dict(arm=job['stage'],cohort=job['scene']['cohort'],scene_seed=job['scene']['seed'],episode_id=job['scene']['episode_id'],inference_seed=job['inference_seed'],job_id=job['job_id'],steps=summary['control_steps'],replans=summary['replans'],grasp=summary['ever_grasped'],held15=metrics['same_fruit_held_at_least_15_steps'],grasp_by_step300=metrics['grasp_by_step300'],held15_qualified_by_step300=metrics['held15_qualified_by_step300'],any_detach=metrics['any_detach'],grasp_and_any_detach=metrics['grasp_and_any_detach'],success=summary['success'],first_grasp_control_step=metrics['first_grasp_control_step'],first_detach_control_step=metrics['first_detach_control_step'],max_same_fruit_held_steps=metrics['max_same_fruit_held_steps'],ik_failed_steps=metrics['ik_failed_steps'],ik_failed_rate=metrics['ik_failed_rate'],action_clipped_steps=metrics['clipping']['union_steps'],action_clipped_rate=metrics['clipping']['union_rate'],dwell_timeouts=metrics['advancement']['dwell_timeouts'],planned_pregrasp_min_tcp_distance_m=distance(metrics,'minimum_tcp_to_planned_apple_before_first_grasp'),nearest_pregrasp_min_tcp_distance_m=distance(metrics,'minimum_tcp_to_nearest_apple_before_first_grasp'),held_apple_id_at_first_success=metrics['held_apple_id_at_first_success'],release_observed_strictly_before_success=metrics['release_observed_strictly_before_success'],release_observed_by_success_boundary=metrics['release_observed_by_success_boundary'],observed_release_count=metrics['observed_held_to_none_release_count'],same_grasped_fruit_detach_verified=metrics['same_grasped_fruit_detach_verified'],first_chunk_teacher_error=metrics['first_chunk_teacher_error'],source_directory=str(directory),completed_sha256=sha(directory/'completed.json'),raw_step_threshold_reconstruction=raw_check)
    return row

def aggregate(rows):
    successes=[r for r in rows if r['success']]
    return dict(episodes=len(rows),independent_scenes=len(set(r['scene_seed'] for r in rows)),counts={m:sum(r[m] for r in rows) for m in BINARY},success_boundary=dict(successes=len(successes),still_held=sum(r['held_apple_id_at_first_success'] is not None for r in successes),no_held_id=sum(r['held_apple_id_at_first_success'] is None for r in successes),observed_release_strictly_before=sum(r['release_observed_strictly_before_success'] is True for r in successes),observed_release_by_boundary=sum(r['release_observed_by_success_boundary'] is True for r in successes)),descriptive={m:stats(r[m] for r in rows) for m in ['steps','replans','first_grasp_control_step','max_same_fruit_held_steps','ik_failed_rate','action_clipped_rate','dwell_timeouts','planned_pregrasp_min_tcp_distance_m','nearest_pregrasp_min_tcp_distance_m']},control_weighted_rates=dict(steps=sum(r['steps'] for r in rows),IK=sum(r['ik_failed_steps'] for r in rows)/sum(r['steps'] for r in rows),action_clipping=sum(r['action_clipped_steps'] for r in rows)/sum(r['steps'] for r in rows)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='closed_loop_final');ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:
        assert pair([True,False,True,False],[True,True,False,False])==dict(pairs=4,A_only=1,B_only=1,both=1,neither=1,B_minus_A_success_count=0)
        assert stats([None,1,3])==dict(count=2,mean=2.0,median=2.0,min=1.0,max=3.0)
        print('CPU helper tests PASS; no rollout outcomes read');return
    # Fail closed on partial queues; no ongoing heldout outcomes are summarized for tuning.
    assert read(LOOP/'queue_status.json')['status']=='COMPLETE','Primary queue incomplete'
    protocol=read(LOOP/'protocol.json');digest=sha(LOOP/'protocol.json');endpoints=read(LOOP/'endpoint_checkpoints.json')
    assert protocol['primary_endpoint_step']==8000 and protocol['gt_episode_count']==20 and protocol['model_episode_count']==64
    assert sha(ROOT/'selection.json')==protocol['selection_sha256']
    for path,expected in protocol['source_sha256'].items():assert sha(path)==expected,('pinned source changed',path)
    rows=[]
    for stage,n in [('GT',20),('A',32),('B',32)]:
        finished=read(LOOP/f'stage_{stage}_complete.json');assert finished['status']=='COMPLETE' and finished['episodes']==n and finished['protocol_sha256']==digest
        jobs=protocol['jobs'][stage];assert len(jobs)==n
        rows.extend(collect(job,digest,endpoints) for job in jobs)
    summary={}
    for arm in ['GT','A','B']:
        for cohort in ['dev','heldout']:
            group=[r for r in rows if r['arm']==arm and r['cohort']==cohort]
            summary[f'{arm}/{cohort}/all']=aggregate(group)
            if arm!='GT':
                for noise in sorted(set(r['inference_seed'] for r in group)):
                    summary[f'{arm}/{cohort}/seed{noise}']=aggregate([r for r in group if r['inference_seed']==noise])
    paired={};by_scene=[]
    for cohort,noise_seeds in [('dev',[42]),('heldout',[42,43])]:
        for noise in noise_seeds:
            aa={r['scene_seed']:r for r in rows if r['arm']=='A' and r['cohort']==cohort and r['inference_seed']==noise};bb={r['scene_seed']:r for r in rows if r['arm']=='B' and r['cohort']==cohort and r['inference_seed']==noise};assert set(aa)==set(bb)
            ids=sorted(aa);paired[f'{cohort}/seed{noise}']={m:pair([aa[s][m] for s in ids],[bb[s][m] for s in ids]) for m in BINARY}
        for scene in sorted(set(r['scene_seed'] for r in rows if r['cohort']==cohort)):
            entry=dict(cohort=cohort,scene_seed=scene,by_arm={})
            for arm in ['A','B']:
                rr=[r for r in rows if r['arm']==arm and r['cohort']==cohort and r['scene_seed']==scene]
                assert sorted(r['inference_seed'] for r in rr)==noise_seeds
                entry['by_arm'][arm]={m:dict(by_inference_seed={str(r['inference_seed']):r[m] for r in rr},all_seeds=all(r[m] for r in rr),any_seed=any(r[m] for r in rr)) for m in BINARY}
            by_scene.append(entry)
    robust=[x for x in by_scene if x['cohort']=='heldout']
    paired['heldout/both_seeds_scene_unit']={m:pair([x['by_arm']['A'][m]['all_seeds'] for x in robust],[x['by_arm']['B'][m]['all_seeds'] for x in robust]) for m in BINARY}
    robust_counts={arm:{m:dict(both_seeds=sum(x['by_arm'][arm][m]['all_seeds'] for x in robust),at_least_one_seed=sum(x['by_arm'][arm][m]['any_seed'] for x in robust),scenes=len(robust)) for m in BINARY} for arm in ['A','B']}
    failures=[dict(scene_seed=r['scene_seed'],episode_id=r['episode_id'],cohort=r['cohort']) for r in rows if r['arm']=='GT' and not r['success']]
    result=dict(schema='orchard_ab_closed_loop_independent_summary_v1',status='PASS',primary_endpoint_step=8000,protocol_sha256=digest,script_sha256=sha(__file__),all84_completed_hashes_valid=True,all_raw_step_thresholds_and_event_counts_recomputed=True,all_observer_reset_summary_checks_pass=True,all_primary_model_seed_deliveries_match=True,GT_failures_retained=failures,aggregate=summary,paired_binary=paired,heldout_robust_counts=robust_counts,per_scene_noise_robustness=by_scene,episodes=rows,limits=['dev8 and heldout12 are separate curated panels; do not treat the24 heldout repeated rollouts as24 independent scenes.','Two generation seeds are not independent training replicates; one training seed/run per arm.','B differs by phase quotas, episode balancing and sampling with replacement.','Any-detach is an aggregate count; grasp-and-any-detach does not verify the same fruit detached.','Benchmark bucket success may occur while fruit remains held. Report held ID and observed release separately; no strict release-completion claim.','Held15 means same fruit held at15 consecutive post-control boundaries, without an imposed hold gate; internal substeps are not independently observed.','Pre-grasp nearest/planned distances exclude attached-fruit proximity after first grasp.','Ground-truth replay failures remain in all paired model totals. No scene exclusions or endpoint selection from model outcomes.','All outcomes are analyzed only after the fixed primary queue completes; no online heldout tuning.'])
    out=HERE/args.name;out.mkdir(exist_ok=False);write(out/'closed_loop_summary.json',result)
    with (out/'episodes.csv').open('x',newline='') as f:
        fields=[k for k in rows[0] if k not in ['first_chunk_teacher_error','raw_step_threshold_reconstruction']];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
    lines=['Fixed step8000 A/B closed-loop comparison, all84 runs hash-verified.','GT20; dev8 perarm seed42; heldout12 perarm seeds42/43. Heldout repeated trajectories do not increase independent scene count.','']
    for key,value in summary.items():
        if key.endswith('/all') and not key.startswith('GT/'):continue
        lines.append(key+': '+json.dumps(value['counts'])+f" / {value['episodes']} episodes")
        lines.append('  Success boundary: '+json.dumps(value['success_boundary']))
    lines.extend(['','Paired binary outcomes (B_only means B succeeds while A fails):'])
    lines.extend(key+': '+json.dumps(value) for key,value in paired.items())
    lines.extend(['','Heldout scene robustness (12 scenes, not24 independent trials):',json.dumps(robust_counts)])
    lines.extend(['','Limits:']+result['limits']);(out/'closed_loop_report.txt').write_text('\n'.join(lines)+'\n')
    (out/'summarize_closed_loop_source.py').write_text(Path(__file__).read_text())
    print('PASS',out)

if __name__=='__main__':main()
