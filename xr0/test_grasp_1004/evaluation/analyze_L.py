"""Paired current-run L/B evaluation; each scheduler remains a separate comparison."""
import argparse
import json
import statistics
from pathlib import Path
import evaluate_grasp as e

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--label',default='final');args=parser.parse_args()
 R=e.ROOT;L=R/'L8000';protocol=json.loads((L/'protocol.json').read_text());rows=[];missing=[]
 for stage,jobs in protocol['jobs'].items():
  for job in jobs:
   for arm,root,prefix in [('L',L,'L8000'),('B',R,'B8000')]:
    folder=root/'rollouts'/job['job_id'].replace('L8000',prefix)
    if not (folder/'completed.json').exists():missing.append(str(folder));continue
    s=json.loads((folder/'summary.json').read_text());m=json.loads((folder/'metrics.json').read_text());steps=e.read_lines(folder/'steps.jsonl');chunks=e.read_lines(folder/'chunks.jsonl');strict=s['strict_placement']
    rows.append(dict(arm=arm,scheduler=job['processed_targets'],seed=job['scene']['seed'],inference_seed=job['inference_seed'],cohort=job['cohort'],held_any=s['ever_grasped'],held15=m['same_fruit_held_at_least_15_steps'],held15_by300=m['held15_qualified_by_step300'],first_grasp=next((r['control_step'] for r in steps if r['grasp_after']['held_apple_id'] is not None),None),same_fruit_detach=any(f['first_detach_after_grasp_step'] is not None for f in strict['fruit_chains'].values()),released=bool(strict['release_events']),valid_chain_release=any(f['valid_chain'] for f in strict['release_events']),strict_success=s['strict_success'],legacy_success_ever=s['legacy_success_ever'],legacy_success_final=s['success'],replans=s['replans'],prediction_seconds=sum(c['prediction_seconds'] for c in chunks),steps=s['control_steps'],ik_fail_steps=s['ik_failed_steps'],dwell_timeouts=s['dwell_timeouts'],branch_break_count=s['max_branch_break_count'],first_negative_intent=next((r['control_step'] for r in steps if r['grasp_after']['gripper_intent']<0),None),path=str(folder)))
 groups={};pairs=[]
 for scheduler in [30,5]:
  for arm in ['B','L']:
   rr=[r for r in rows if r['cohort']=='development' and r['scheduler']==scheduler and r['arm']==arm]
   if not rr:continue
   groups[f'{arm}_targets{scheduler}']=dict(n=len(rr),**{k:sum(r[k] for r in rr) for k in ['held_any','held15','held15_by300','same_fruit_detach','released','valid_chain_release','strict_success','legacy_success_ever','ik_fail_steps','dwell_timeouts','branch_break_count']},mean_replans=statistics.mean(r['replans'] for r in rr),mean_prediction_seconds=statistics.mean(r['prediction_seconds'] for r in rr),first_grasp_steps=[r['first_grasp'] for r in rr if r['first_grasp'] is not None])
  for seed in sorted({r['seed'] for r in rows if r['cohort']=='development' and r['scheduler']==scheduler}):
   arms={r['arm']:r for r in rows if r['cohort']=='development' and r['scheduler']==scheduler and r['seed']==seed}
   if len(arms)==2:pairs.append(dict(seed=seed,scheduler=scheduler,B=arms['B'],L=arms['L'],held15_changed=int(arms['L']['held15'])-int(arms['B']['held15']),strict_changed=int(arms['L']['strict_success'])-int(arms['B']['strict_success'])))
 out=L/'analysis'/args.label;out.mkdir(parents=True,exist_ok=False)
 result=dict(status='COMPLETE' if not missing else 'PARTIAL',protocol_sha256=e.sha(L/'protocol.json'),baseline_protocol_sha256=e.sha(R/'protocol.json'),analysis_source_sha256=e.sha(Path(__file__)),groups=groups,pairs=pairs,auxiliary_regression_rows=[r for r in rows if r['cohort']=='historical_success_regression'],missing=missing,rows=rows,limitations=['Both arms have one training seed; two schedulers are distinct strata, not independent repeats.','Development8 contains2training scenes and6validation scenes already exposed during development.','This comparison changes L loss weights while preserving B sampling table and budget; absolute B outcomes can differ from historical runs due to closed-loop physics.','Strict success observes the same fruit held15, detached, released, and60consecutive bucket/notheld boundary samples within900steps.'])
 e.save_new(out/'paired_results.json',result)
 lines=['L8000 vs current B8000, matching scheduler: '+result['status'],'arm/scheduler | n | held15 | samefruit detach | releases | strict60 | legacy ever | mean replans']
 for key,g in groups.items():lines.append(f'{key} | {g["n"]} | {g["held15"]} | {g["same_fruit_detach"]} | {g["released"]} | {g["strict_success"]} | {g["legacy_success_ever"]} | {g["mean_replans"]:.2f}')
 (out/'report.txt').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(status=result['status'],groups=groups,output=str(out)),indent=2))
if __name__=='__main__':main()
