"""CPU-only independent dev8 train-best comparison; primary and heldout stay fixed."""
import csv, hashlib, importlib.util, json, sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
DEP=ROOT.parent/'analysis_1004/summarize_closed_loop.py'
spec=importlib.util.spec_from_file_location('primary_independent_audit',DEP);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
read=audit.read;sha=audit.sha
FIELDS=['grasp','held15','grasp_by_step300','held15_qualified_by_step300','any_detach','grasp_and_any_detach','success']

def secondary_row(job,endpoint,protocol_sha):
 directory=ROOT/'rollouts'/job['job_id'];done=read(directory/'completed.json')
 assert done['status']=='COMPLETE' and done['protocol_sha256']==protocol_sha and done['checkpoint_sha256']==endpoint['sha256']
 assert done['job_id']==job['job_id'] and done['inference_seed']==42
 assert done['annotation_sha256']==job['scene']['annotation_sha256']==sha(job['scene']['annotation'])
 for filename,digest in done['files_sha256'].items():assert sha(directory/filename)==digest
 s=read(directory/'summary.json');m=read(directory/'metrics.json');reset=read(directory/'scene_check.json');seed=read(directory/'seed_delivery.json')
 assert s['checkpoint_step']==m['checkpoint_step']==4000 and s['comparison_role']==m['comparison_role']=='secondary_train_best'
 assert s['cohort']==m['cohort']=='dev' and s['seed']==job['scene']['seed'] and s['episode_id']==job['scene']['episode_id']
 assert s['inference_seed']==m['inference_seed']==42
 assert s['termination']!='error' and s['scene_check_pass'] and reset['state_and_identity_pass']
 assert reset['current_gt_pairing']['exact_match'] and reset['current_gt_pairing']['gt_runtime_valid']
 assert seed['all_calls_match'] and len(seed['calls'])==s['replans'] and all(c['actual_inference_seed']==42 for c in seed['calls'])
 assert m['observer_consistency']['mismatch_fields_total']==0 and m['summary_consistency']['mismatch_count']==0
 assert m['actual_grasp_event']==s['ever_grasped']==done['grasp'] and m['same_fruit_held_at_least_15_steps']==done['held15'] and s['success']==done['success']
 steps=[json.loads(x) for x in (directory/'steps.jsonl').read_text().splitlines()]
 validation=audit.independently_reconstruct(steps,s,m,job)
 return dict(arm='B',checkpoint_step=4000,comparison_role='secondary_train_best',cohort='dev',scene_seed=s['seed'],episode_id=s['episode_id'],inference_seed=42,job_id=job['job_id'],steps=s['control_steps'],replans=s['replans'],grasp=s['ever_grasped'],held15=m['same_fruit_held_at_least_15_steps'],grasp_by_step300=m['grasp_by_step300'],held15_qualified_by_step300=m['held15_qualified_by_step300'],any_detach=m['any_detach'],grasp_and_any_detach=m['grasp_and_any_detach'],success=s['success'],first_grasp_control_step=m['first_grasp_control_step'],first_detach_control_step=m['first_detach_control_step'],max_same_fruit_held_steps=m['max_same_fruit_held_steps'],ik_failed_steps=m['ik_failed_steps'],ik_failed_rate=m['ik_failed_rate'],action_clipped_steps=m['clipping']['union_steps'],action_clipped_rate=m['clipping']['union_rate'],dwell_timeouts=m['advancement']['dwell_timeouts'],planned_pregrasp_min_tcp_distance_m=audit.distance(m,'minimum_tcp_to_planned_apple_before_first_grasp'),nearest_pregrasp_min_tcp_distance_m=audit.distance(m,'minimum_tcp_to_nearest_apple_before_first_grasp'),held_apple_id_at_first_success=m['held_apple_id_at_first_success'],release_observed_strictly_before_success=m['release_observed_strictly_before_success'],release_observed_by_success_boundary=m['release_observed_by_success_boundary'],observed_release_count=m['observed_held_to_none_release_count'],same_grasped_fruit_detach_verified=None,first_chunk_teacher_error=m['first_chunk_teacher_error'],source_directory=str(directory),completed_sha256=sha(directory/'completed.json'),raw_step_threshold_reconstruction=validation)

def main():
 assert read(ROOT/'queue_status.json')['status']=='COMPLETE'
 marker=read(ROOT/'stage_B_trainbest_complete.json');assert marker['status']=='COMPLETE' and marker['episodes']==8
 protocol=read(ROOT/'protocol.json');digest=sha(ROOT/'protocol.json');assert marker['protocol_sha256']==digest
 endpoint=read(ROOT/'secondary_endpoint_B.json');assert endpoint['step']==4000 and sha(endpoint['path'])==endpoint['sha256']==marker['checkpoint_sha256']
 assert sha(ROOT.parent/'arms/B/best_checkpoint.json')==endpoint['train_best_record_sha256']
 for path,value in protocol['source_sha256'].items():assert sha(path)==value
 endpoints=read(ROOT/'endpoint_checkpoints.json');rows=[]
 for arm in ['A','B']:
  jobs=[j for j in protocol['jobs'][arm] if j['scene']['cohort']=='dev'];assert len(jobs)==8
  for j in jobs:
   r=audit.collect(j,digest,endpoints);r.update(checkpoint_step=8000,comparison_role='primary_endpoint');rows.append(r)
 for j in [j for j in protocol['jobs']['B'] if j['scene']['cohort']=='dev']:
  secondary=dict(j,checkpoint_step=4000,provider='B_trainbest_step4000_rng42',job_id=f'B_trainbest_step4000_rng42_reach-conditioned_{j["scene"]["seed"]}',comparison_role='secondary_train_best')
  rows.append(secondary_row(secondary,endpoint,digest))
 groups={f'{a}{s}':[r for r in rows if r['arm']==a and r['checkpoint_step']==s] for a,s in [('A',8000),('B',8000),('B',4000)]}
 by={k:{r['scene_seed']:r for r in rr} for k,rr in groups.items()};ids=sorted(by['B8000']);assert all(sorted(v)==ids for v in by.values())
 pair={field:dict(scenes=8,B4000_only=sum(by['B4000'][s][field] and not by['B8000'][s][field] for s in ids),B8000_only=sum(by['B8000'][s][field] and not by['B4000'][s][field] for s in ids),both=sum(by['B4000'][s][field] and by['B8000'][s][field] for s in ids),neither=sum(not by['B4000'][s][field] and not by['B8000'][s][field] for s in ids)) for field in FIELDS}
 result=dict(status='PASS',schema='orchard_ab_secondary_train_best_dev8_v1',protocol_sha256=digest,script_sha256=sha(__file__),independent_auditor_sha256=sha(DEP),checkpoint=endpoint,all24_primary_secondary_dev_hashes_and_raw_steps_valid=True,aggregate={k:audit.aggregate(v) for k,v in groups.items()},paired_B4000_B8000=pair,episodes=rows,limits=['Secondary B4000 was selected by TRAIN-only best rule before closed-loop testing and separately authorized before primary queue completed.','Only fixed dev8 seed42 was evaluated; no heldout best-checkpoint testing or checkpoint replacement.','Primary remains A8000 vs B8000. A8000 is reused because its train-only best equals the primary endpoint.','Same unchanged full30 reach loop, 900-step budget, control limits, benchmark_assist and fixed reset.','Held15 is consecutive control boundaries, not imposed pose holding or physics-substep proof.','Any-detach is aggregate; benchmark bucket success is not inherently an explicit-release completion.'])
 out=ROOT/'secondary_train_best_analysis';out.mkdir(exist_ok=False)
 with (out/'summary.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 lines=['Secondary TRAIN-best B4000: fixed dev8 seed42; primary endpoints remain8000.']
 for key,group in result['aggregate'].items():lines.append(key+': '+json.dumps(group['counts'])+' /8')
 lines.extend(['Paired B4000 vs B8000: '+json.dumps(pair),'']+result['limits'])
 (out/'report.txt').write_text('\n'.join(lines)+'\n')
 fields=[k for k in rows[0] if k not in ['first_chunk_teacher_error','raw_step_threshold_reconstruction']]
 with (out/'episodes.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
 print(json.dumps(dict(status='PASS',path=str(out),counts={k:v['counts'] for k,v in result['aggregate'].items()},paired=pair)))
if __name__=='__main__':main()
