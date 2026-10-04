"""Additive CPU-only comparison; never reruns physics or edits protocol sources."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
D0=R.parent
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'cpu_only':True,'simulation_rerun':False,'source_sha256':sha(Path(__file__)),'comparisons':[]}
for seed in [2014362,2014461]:
 case={'seed':seed}
 for name,folder,gpath in [('D0',D0/f'rollouts/B8000_targets30_rng42_reach-conditioned_{seed}',D0/f'analysis/palm_decomposition/B30_{seed}_steps.jsonl'),('D1',R/f'rollouts/B8000_pos5mm_commit30_rng42_reach-conditioned_{seed}',R/f'analysis/palm_decomposition/B_pos5mm_{seed}_steps.jsonl')]:
  rr=rows(folder/'steps.jsonl');gg=rows(gpath);a0=np.asarray(rr[0]['grasp_before']['planned_apple']['world_position']);by={r['control_step']:r for r in rr}
  def small(g):
   r=by[g['step']];a=np.asarray(r['grasp_after']['planned_apple']['world_position'])
   return dict(step=g['step'],chunk=g['chunk_id'],k=g['target_k'],dwell=g['dwell_steps'],reached=g['reached'],advance_reason=g['advance_reason'],actual_local_mm=(np.array(g['actual']['local_xyz_m'])*1000).tolist(),target_local_mm=(np.array(g['predicted_target']['local_xyz_m'])*1000).tolist(),actual_margin_mm=1000*g['actual']['minimum_margin_m'],target_margin_mm=1000*g['predicted_target']['minimum_margin_m'],tracking_mm=1000*g['tracking_position_m'],target_width_mm=1000*g['target_width_m'],actual_width_mm=1000*g['actual_width_m'],negative_intent=g['negative_intent'],both_fingers=g['both_finger_contact'],intent_allows_attach=r['grasp_after']['intent_allows_attach'],geometry_eligible=r['grasp_after']['planned_apple']['geometry_eligible'],would_attach_now=r['grasp_after']['would_attach_now'],ik_failed=g['ik_failed'],action_clipped=g['action_clipped'],solver_position_residual_mm=1000*r['info']['ik_error']['position_m'],apple_world_position_m=a.tolist(),apple_delta_from_reset_mm=((a-a0)*1000).tolist(),apple_distance_from_reset_mm=float(np.linalg.norm(a-a0)*1000))
  both=[g for g in gg if g['negative_intent'] and g['both_finger_contact']]
  inside=[g for g in both if g['predicted_target']['inside_strict']]
  candidates=[r for r in rr if .005<r['info']['ik_error']['position_m']<=.01 and r['info']['ik_error']['rotation_rad']<=.08]
  critical=[small(g) for g in gg if g['chunk_id']==0 and g['target_k'] in ([17,19,20,21] if seed==2014362 else [15])]
  case[name]=dict(input_sha256=sha(folder/'steps.jsonl'),first_negative=next((g['step'] for g in gg if g['negative_intent']),None),first_both_contact=next((g['step'] for g in gg if g['both_finger_contact']),None),first_replan_step=next((g['step'] for g in gg if g['chunk_id']==1),None),negative_both_samples=len(both),actual_inside_negative_both=sum(g['actual']['inside_strict'] for g in both),target_inside_negative_both=len(inside),target_inside_actual_outside_reached=sum(g['reached'] and not g['actual']['inside_strict'] for g in inside),best_actual_negative_both=small(max(both,key=lambda g:g['actual']['minimum_margin_m'])),critical_first_chunk_targets=critical,ik_5to10_accepted_rotation_count=len(candidates),ik_5to10_first_step=candidates[0]['control_step'] if candidates else None,ik_5to10_last_step=candidates[-1]['control_step'] if candidates else None)
 result['comparisons'].append(case)
# All four model traces: do added rejection candidates precede first contact/grasp?
result['rejection_timing']=[]
for seed in [2014461,2014362,2013322,2012485]:
 folder=R/f'rollouts/B8000_pos5mm_commit30_rng42_reach-conditioned_{seed}';rr=rows(folder/'steps.jsonl');cs=[r for r in rr if .005<r['info']['ik_error']['position_m']<=.01 and r['info']['ik_error']['rotation_rad']<=.08];held=next((r['control_step'] for r in rr if r['grasp_after']['held_apple_id'] is not None),None)
 result['rejection_timing'].append(dict(seed=seed,count=len(cs),first_step=cs[0]['control_step'] if cs else None,last_step=cs[-1]['control_step'] if cs else None,first_grasp=held,prior_to_first_grasp=sum(r['control_step']<held for r in cs) if held is not None else None,arm_command_held_count=sum(max(abs(v) for v in r['joint_command_delta']['delta'][:7])<=1e-12 for r in cs)))
out=R/'analysis/final';(out/'geometry_timing_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
for c in result['comparisons']:
 print(c['seed'])
 for label in ['D0','D1']:
  z=c[label];print(label,'firstreplan',z['first_replan_step'],'critical',[(x['step'],x['k'],round(x['actual_margin_mm'],3),round(x['target_margin_mm'],3),round(x['tracking_mm'],3),round(x['apple_distance_from_reset_mm'],3),x['reached']) for x in z['critical_first_chunk_targets']])
print('rejections',result['rejection_timing'])
