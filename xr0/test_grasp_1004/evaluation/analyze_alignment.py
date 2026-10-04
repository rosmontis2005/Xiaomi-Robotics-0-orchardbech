"""Supplemental source-index alignment and event analysis, without simulator import."""
import argparse
import json
from pathlib import Path
import evaluate_grasp as e

def rows(path):return e.read_lines(path/'steps.jsonl')
def target_map(steps):
 result={}
 for r in steps:
  if r.get('terminal_hold'):continue
  key=r['gt_target_index'];value={k:r[k] for k in ['target_position','target_rotation','target_width']}
  if key in result:assert result[key]==value
  result[key]=value
 return result

def events(steps):
 return dict(first_held=next((r['control_step'] for r in steps if r['grasp_after']['held_apple_id'] is not None),None),first_detach=next((r['control_step'] for r in steps if r['info']['apple_detached_count']>0),None),releases=[dict(step=r['control_step'],apple_id=r['grasp_before']['held_apple_id']) for r in steps if r['grasp_before']['held_apple_id'] is not None and r['grasp_after']['held_apple_id'] is None],first_legacy_success=next((r['control_step'] for r in steps if r['info']['success']),None),ik_fail_count=sum(r['info']['ik_failed'] for r in steps),dwell_timeout_count=sum(r['advance_reason']=='maximum_dwell' for r in steps),branch_break_count=max([r['info']['branch_break_count'] for r in steps],default=0))

def main():
 p=argparse.ArgumentParser();p.add_argument('--label',default='final');args=p.parse_args()
 root=e.ROOT;protocol=json.loads(e.PROTOCOL.read_text());out=root/'analysis'/args.label;out.mkdir(parents=True,exist_ok=True)
 alignments=[];model_rows=[]
 for job in protocol['jobs']['GT30']:
  seed=job['scene']['seed'];left=root/'rollouts'/job['job_id'];right=root/'rollouts'/f'GT_targets5_reach-conditioned_{seed}'
  if not (left/'completed.json').exists() or not (right/'completed.json').exists():continue
  a,b=rows(left),rows(right);ma,mb=target_map(a),target_map(b);common=sorted(set(ma)&set(mb))
  assert all(ma[k]==mb[k] for k in common),'GT absolute goal stream changed'
  sa=json.loads((left/'summary.json').read_text());sb=json.loads((right/'summary.json').read_text())
  alignments.append(dict(seed=seed,GT30_steps=len(a),GT5_steps=len(b),GT30_unique_indices=len(ma),GT5_unique_indices=len(mb),common_indices=len(common),same_visited_index_set=set(ma)==set(mb),absolute_goals_by_source_index_exact=True,GT30_strict_success=sa['strict_success'],GT5_strict_success=sb['strict_success'],GT30_events=events(a),GT5_events=events(b),GT30_strict=sa['strict_placement']['strict_success_events'],GT5_strict=sb['strict_placement']['strict_success_events']))
 for stage in ['B30','B5','regression_B30']:
  for job in protocol['jobs'][stage]:
   folder=root/'rollouts'/job['job_id']
   if not (folder/'completed.json').exists():continue
   s=json.loads((folder/'summary.json').read_text());m=json.loads((folder/'metrics.json').read_text());trace=rows(folder);chunks=e.read_lines(folder/'chunks.jsonl')
   regression=m['control_regression']
   model_rows.append(dict(stage=stage,seed=job['scene']['seed'],inference_seed=job['inference_seed'],held_any=s['ever_grasped'],held15=m['same_fruit_held_at_least_15_steps'],events=events(trace),strict=s['strict_placement'],replans=s['replans'],control_steps=s['control_steps'],mean_control_steps_per_chunk=s['control_steps']/s['replans'] if s['replans'] else None,completed_targets=sum(r['advance'] for r in trace),prediction_seconds=sum(c['prediction_seconds'] for c in chunks),physics_sim_seconds=trace[-1]['sim_time'],first_chunk_target_values_exact_to_history=None,control_regression=regression))
   if stage in ['B30','regression_B30']:
    old=e.OLD/'rollouts'/f'B_step8000_rng{job["inference_seed"]}_reach-conditioned_{job["scene"]["seed"]}'
    oldchunks=e.read_lines(old/'chunks.jsonl');oldinitial=json.loads((old/'initial.json').read_text())['obs'];initial=json.loads((folder/'initial.json').read_text())['obs']
    model_rows[-1]['initial_observation_exact_to_history']=oldinitial==initial
    model_rows[-1]['first_chunk_target_values_exact_to_history']=all(chunks[0][k]==oldchunks[0][k] for k in ['target_positions','target_rotations','target_widths'])
    model_rows[-1]['historical_events']=events(rows(old))
 result=dict(protocol_sha256=e.sha(e.PROTOCOL),analysis_source_sha256=e.sha(Path(__file__)),gt_pairs=alignments,model_rows=model_rows,interpretation='Source-index GT goal identity is a separate assertion from wall-step physical replay equality. CUDA contact dynamics can shift reach threshold crossings. Current-run matching-scheduler B is the baseline for L; model target differences are never hidden under GT physical tolerances.')
 e.save_new(out/'alignment_and_events.json',result)
 print(json.dumps(dict(gt_pairs=len(alignments),model_episodes=len(model_rows),all_gt_source_index_targets_exact=all(r['absolute_goals_by_source_index_exact'] for r in alignments),output=str(out/'alignment_and_events.json'))))
if __name__=='__main__':main()
