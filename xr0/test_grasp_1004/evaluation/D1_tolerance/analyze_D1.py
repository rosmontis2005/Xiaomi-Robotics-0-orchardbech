"""Read-only fixed D1 pilot comparison; thresholds are explicitly different."""
import argparse,json,statistics
from pathlib import Path
import evaluate_D1 as e

def geometry_stats(trace):
 import numpy as np
 from scipy.spatial.transform import Rotation
 first=next((r['control_step'] for r in trace if r['grasp_after']['held_apple_id'] is not None),None)
 candidates=[];all_rows=[]
 for r in trace:
  if first is not None and r['control_step']>=first:continue
  snap=r['grasp_after'];apple=snap['planned_apple'];ra=Rotation.from_quat(r['measured_quaternion']);rh=Rotation.from_quat(snap['hand_quaternion_world']);t=ra.inv().apply(np.asarray(snap['hand_position_world'])-r['measured_position']);rel=ra.inv()*rh;rt=Rotation.from_matrix(r['target_rotation'])
  desired=(rt*rel).inv().apply(np.asarray(apple['world_position'])-(np.asarray(r['target_position'])+rt.apply(t)))
  def margin(x):return float(min(.025-abs(x[0]),.04-abs(x[1]),x[2]-.065,.125-x[2]))
  q=dict(step=r['control_step'],target_k=r['target_k'],chunk_id=r['chunk_id'],actual_margin_m=margin(apple['hand_local_position']),target_margin_m=margin(desired),actual_local_xyz_m=apple['hand_local_position'],target_local_xyz_m=desired.tolist(),reached=r['reached'],tracking_position_m=r['position_error_m'],target_width_m=r['target_width'],actual_width_m=r['measured_width'],negative_intent=snap['gripper_intent']<0,both_contact=apple['both_fingers_contact'])
  all_rows.append(q)
  if q['negative_intent'] and q['both_contact']:candidates.append(q)
 return dict(pregrasp_samples=len(all_rows),negative_and_both_samples=len(candidates),best_actual_while_negative_and_both=max(candidates,key=lambda q:q['actual_margin_m']) if candidates else None,best_target_while_negative_and_both=max(candidates,key=lambda q:q['target_margin_m']) if candidates else None,target_inside_actual_outside_samples=sum(q['target_margin_m']>0 and q['actual_margin_m']<=0 for q in candidates),target_inside_actual_outside_marked_reached=sum(q['target_margin_m']>0 and q['actual_margin_m']<=0 and q['reached'] for q in candidates))

def main():
 p=argparse.ArgumentParser();p.add_argument('--label',default='final');args=p.parse_args()
 protocol=json.loads(e.PROTOCOL.read_text());rows=[];missing=[]
 for stage,jobs in protocol['jobs'].items():
  for job in jobs:
   folder=e.ROOT/'rollouts'/job['job_id']
   if not (folder/'completed.json').exists():missing.append(job['job_id']);continue
   s=e.completion(job);m=json.loads((folder/'metrics.json').read_text());steps=e.read_lines(folder/'steps.jsonl')
   provider='GT_targets30' if job['is_gt'] else 'B8000_targets30_rng42';old=e.D0/'rollouts'/f'{provider}_reach-conditioned_{job["scene"]["seed"]}'
   bs=json.loads((old/'summary.json').read_text());bm=json.loads((old/'metrics.json').read_text());bsteps=e.read_lines(old/'steps.jsonl')
   def metrics(summary,met,trace):
    residuals=[r['info']['ik_error']['position_m'] for r in trace]
    candidates=[r for r in trace if .005<r['info']['ik_error']['position_m']<=.01 and r['info']['ik_error']['rotation_rad']<=.08]
    return dict(held_any=summary['ever_grasped'],held15=met['same_fruit_held_at_least_15_steps'],first_grasp=next((r['control_step'] for r in trace if r['grasp_after']['held_apple_id'] is not None),None),strict_success=summary['strict_success'],legacy_success_ever=summary['legacy_success_ever'],same_fruit_detach=any(x['first_detach_after_grasp_step'] is not None for x in summary['strict_placement']['fruit_chains'].values()),releases=len(summary['strict_placement']['release_events']),steps=summary['control_steps'],dwell_timeouts=summary['dwell_timeouts'],ik_failed_recorded=summary['ik_failed_steps'],solver_position_residual_max_m=max(residuals),solver_position_residual_mean_m=statistics.mean(residuals),solver_residual_5to10mm_rotation_acceptable=len(candidates),candidate_arm_command_held=sum(all(abs(v)<=1e-12 for v in r['joint_command_delta']['delta'][:7]) for r in candidates),branch_breaks=summary['max_branch_break_count'],replans=summary['replans'],pregrasp_geometry=geometry_stats(trace))
   item=dict(stage=stage,seed=job['scene']['seed'],D1=metrics(s,m,steps),D0=metrics(bs,bm,bsteps),path=str(folder),baseline=str(old))
   if job['is_gt']:
    def goals(trace):
     result={}
     for r in trace:
      if r.get('terminal_hold'):continue
      result[r['gt_target_index']]=[r['target_position'],r['target_rotation'],r['target_width']]
     return result
    a,b=goals(steps),goals(bsteps);common=set(a)&set(b);assert all(a[k]==b[k] for k in common)
    item['source_goal_alignment']=dict(D1_unique_targets=len(a),D0_unique_targets=len(b),common=len(common),absolute_targets_exact_by_source_index=True)
   rows.append(item)
 groups={}
 for stage in protocol['queue_order']:
  rr=[x for x in rows if x['stage']==stage]
  if not rr:continue
  groups[stage]=dict(n=len(rr),D1={key:sum(r['D1'][key] for r in rr) for key in ['held15','strict_success','legacy_success_ever','same_fruit_detach','releases','dwell_timeouts','ik_failed_recorded','solver_residual_5to10mm_rotation_acceptable','candidate_arm_command_held','branch_breaks']},D0={key:sum(r['D0'][key] for r in rr) for key in ['held15','strict_success','legacy_success_ever','same_fruit_detach','releases','dwell_timeouts','ik_failed_recorded','solver_residual_5to10mm_rotation_acceptable','candidate_arm_command_held','branch_breaks']})
 out=e.ROOT/'analysis'/args.label;out.mkdir(parents=True,exist_ok=False)
 result=dict(status='COMPLETE' if not missing else 'PARTIAL',protocol_sha256=e.sha(e.PROTOCOL),source_sha256=e.sha(Path(__file__)),groups=groups,rows=rows,missing=missing,interpretation=['D1 changes both IK result acceptance and actual reach position threshold10mm to5mm; solver32iterations and objective weights unchanged.','Recorded IK_failed in D0 and D1 uses different thresholds; residual bins and held command counts are reported explicitly.','Ideal FK residual and measured servo target tracking error are distinct.','Four predeclared diagnostic scenes are not an unbiased success-rate estimate; no automatic expansion follows failed GT control.'])
 e.save_new(out/'results.json',result);lines=['D1 fixed5mm acceptance/reach pilot: '+result['status']]
 for stage,g in groups.items():lines.append(f'{stage} n={g["n"]}; D0 held15/strict={g["D0"]["held15"]}/{g["D0"]["strict_success"]}; D1={g["D1"]["held15"]}/{g["D1"]["strict_success"]}; D1 extra5to10mm candidates/held-arm={g["D1"]["solver_residual_5to10mm_rotation_acceptable"]}/{g["D1"]["candidate_arm_command_held"]}')
 (out/'report.txt').write_text('\n'.join(lines+result['interpretation'])+'\n');print(json.dumps(dict(status=result['status'],groups=groups,output=str(out)),indent=2))
if __name__=='__main__':main()
