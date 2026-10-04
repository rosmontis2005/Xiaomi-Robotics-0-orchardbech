"""Read-only analysis of frozen D0 rollouts; may run while queue is incomplete."""
import argparse
import json
from pathlib import Path
import statistics
import time
import evaluate_grasp as evaluation
R=evaluation.ROOT

def distribution(values):
 values=[x for x in values if x is not None]
 return None if not values else dict(n=len(values),mean=statistics.mean(values),min=min(values),max=max(values),median=statistics.median(values))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--label',default='final');args=parser.parse_args()
 protocol=json.loads(evaluation.PROTOCOL.read_text());rows=[];missing=[]
 for stage,jobs in protocol['jobs'].items():
  for job in jobs:
   folder=R/'rollouts'/job['job_id']
   if not (folder/'completed.json').exists():missing.append(job['job_id']);continue
   summary=evaluation.completion(job);metrics=json.loads((folder/'metrics.json').read_text());steps=evaluation.read_lines(folder/'steps.jsonl');chunks=evaluation.read_lines(folder/'chunks.jsonl')
   strict=summary['strict_placement'];first_grasp=next((s['control_step'] for s in steps if s['grasp_after']['held_apple_id'] is not None),None)
   close=next((s['control_step'] for s in steps if s['grasp_after']['gripper_intent']<0),None)
   pre=[s for s in steps if first_grasp is None or s['control_step']<first_grasp]
   allmargin=[]
   for s in pre:
    a=s['grasp_after']['planned_apple'];x,y,z=a['hand_local_position']
    allmargin.append(dict(step=s['control_step'],margin_m=min(.025-abs(x),.04-abs(y),z-.065,.125-z),position=a['hand_local_position'],both_contact=a['both_fingers_contact'],negative_intent=s['grasp_after']['gripper_intent']<0))
   nearest=max(allmargin,key=lambda x:x['margin_m']) if allmargin else None
   pregrasp_replans=sum(c['at_control_step']<(first_grasp or summary['control_steps']+1) for c in chunks) if not job['is_gt'] else 0
   fruit=next(iter(strict['fruit_chains'].values()),{})
   row=dict(stage=stage,cohort=job['cohort'],seed=job['scene']['seed'],inference_seed=job['inference_seed'],processed_targets=job['processed_targets'],held15=metrics['same_fruit_held_at_least_15_steps'],first_grasp_step=first_grasp,first_held15_step=metrics['first_held15_qualification_step'],held15_by300=metrics['held15_qualified_by_step300'],any_detach=summary['max_apple_detached_count']>0,same_grasped_fruit_detach=any(v['first_detach_after_grasp_step'] is not None for v in strict['fruit_chains'].values()),legacy_success_ever=summary['legacy_success_ever'],legacy_success_final=summary['success'],first_legacy_success_step=strict['first_legacy_success_step'],strict_success=summary['strict_success'],strict_success_events=strict['strict_success_events'],release_count=len(strict['release_events']),valid_chain_release_count=sum(e['valid_chain'] for e in strict['release_events']),max_stable_bucket_steps=max([v['max_stable_bucket_steps'] for v in strict['fruit_chains'].values()],default=0),first_negative_intent_step=close,control_steps=summary['control_steps'],replans=summary['replans'],replans_before_first_grasp=pregrasp_replans,prediction_seconds=sum(c['prediction_seconds'] for c in chunks),wall_seconds=summary['wall_seconds'],ik_fail_steps=summary['ik_failed_steps'],clip_steps=summary['action_clipped_steps'],dwell_timeouts=summary['dwell_timeouts'],branch_break_count=summary['max_branch_break_count'],best_pregrasp_planned_palm_margin=nearest,control_regression=metrics['control_regression'],path=str(folder))
   rows.append(row)
 groups={}
 for stage in protocol['queue_order']:
  selected=[r for r in rows if r['stage']==stage]
  if not selected:continue
  counts=['held15','held15_by300','same_grasped_fruit_detach','legacy_success_ever','strict_success']
  groups[stage]=dict(episodes=len(selected),**{key:sum(bool(r[key]) for r in selected) for key in counts},first_grasp_step=distribution(r['first_grasp_step'] for r in selected),replans=distribution(r['replans'] for r in selected),prediction_seconds=distribution(r['prediction_seconds'] for r in selected),release_episodes=sum(r['release_count']>0 for r in selected),branch_breaks=sum(r['branch_break_count'] for r in selected),ik_fail_steps=sum(r['ik_fail_steps'] for r in selected),clip_steps=sum(r['clip_steps'] for r in selected),total_control_steps=sum(r['control_steps'] for r in selected),control_exact_prefixes=sum(r['control_regression'] is not None and r['control_regression']['exact_prefix'] for r in selected))
 pairs=[]
 for scene in json.loads(evaluation.SELECTION.read_text())['development_scenes']:
  arms={r['stage']:r for r in rows if r['seed']==scene['seed'] and r['stage'] in ['B30','B5']}
  if len(arms)==2:pairs.append(dict(seed=scene['seed'],held15_B30=arms['B30']['held15'],held15_B5=arms['B5']['held15'],first_grasp_B30=arms['B30']['first_grasp_step'],first_grasp_B5=arms['B5']['first_grasp_step'],replans_B30=arms['B30']['replans'],replans_B5=arms['B5']['replans'],close_B30=arms['B30']['first_negative_intent_step'],close_B5=arms['B5']['first_negative_intent_step'],strict_B30=arms['B30']['strict_success'],strict_B5=arms['B5']['strict_success']))
 result=dict(status='COMPLETE' if not missing else 'PARTIAL',created_unix=time.time(),protocol_sha256=evaluation.sha(evaluation.PROTOCOL),analysis_source_sha256=evaluation.sha(Path(__file__)),episodes=len(rows),missing=missing,groups=groups,development_model_pairs=pairs,rows=rows,limitations=['Development8 is exposed and includes2training scenes; not a generalization estimate.','Historical success scene2010600 seeds42/43 is a separate regression cohort.','Strict placement is same-fruit control-boundary evidence; internal substeps are not continuously observed.','Original final success and ever success remain distinct from strict release60 success.'])
 target=R/'analysis'/args.label;target.mkdir(parents=True,exist_ok=False)
 evaluation.save_new(target/'results.json',result)
 lines=['D0 deployment diagnostic: '+result['status'],f'Frozen protocol: {result["protocol_sha256"]}',f'Completed {len(rows)}/34 episodes','', 'stage | n | held15 | held15by300 | samefruit detach | legacy ever | strict60 | release episodes | mean replans']
 for stage,g in groups.items():lines.append(f'{stage} | {g["episodes"]} | {g["held15"]} | {g["held15_by300"]} | {g["same_grasped_fruit_detach"]} | {g["legacy_success_ever"]} | {g["strict_success"]} | {g["release_episodes"]} | {g["replans"]["mean"]:.2f}')
 lines+=['','All outcomes and per-step files retained; no scene exclusion.']+result['limitations']
 (target/'report.txt').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(status=result['status'],groups=groups,output=str(target)),indent=2))
if __name__=='__main__':main()
