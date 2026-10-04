"""Read-only contact geometry and closure timing diagnostics."""
import argparse,json
from pathlib import Path
import evaluate_grasp as e

def best_geometry(steps,condition):
 candidates=[]
 for r in steps:
  a=r['grasp_after']['planned_apple'];x,y,z=a['hand_local_position'];margin=min(.025-abs(x),.04-abs(y),z-.065,.125-z)
  if condition(r,a):candidates.append(dict(step=r['control_step'],margin_m=margin,hand_local_position=a['hand_local_position'],both_contact=a['both_fingers_contact'],inside_palm=a['inside_palm_volume'],negative_intent=r['grasp_after']['gripper_intent']<0,command_width=r['grasp_after']['commanded_width_m'],measured_width=r['measured_width'],target_width=r['target_width'],chunk_id=r['chunk_id'],target_k=r['target_k'],target_tracking_position_m=r['position_error_m'],ik_failed=r['info']['ik_failed']))
 return max(candidates,key=lambda r:r['margin_m']) if candidates else None

def main():
 p=argparse.ArgumentParser();p.add_argument('--label',default='final');p.add_argument('--include-L',action='store_true');args=p.parse_args()
 roots=[e.ROOT]+([e.ROOT/'L8000'] if args.include_L else []);rows=[]
 for root in roots:
  protocol=json.loads((root/'protocol.json').read_text())
  for jobs in protocol['jobs'].values():
   for job in jobs:
    if job['is_gt']:continue
    folder=root/'rollouts'/job['job_id']
    if not (folder/'completed.json').exists():continue
    s=json.loads((folder/'summary.json').read_text());steps=e.read_lines(folder/'steps.jsonl');chunks=e.read_lines(folder/'chunks.jsonl')
    first=next((r['control_step'] for r in steps if r['grasp_after']['held_apple_id'] is not None),None)
    approach=[r for r in steps if first is None or r['control_step']<first]
    closes=[r for r in approach if r['grasp_after']['gripper_intent']<0]
    both=[r for r in approach if r['grasp_after']['planned_apple']['both_fingers_contact']]
    inside=[r for r in approach if r['grasp_after']['planned_apple']['inside_palm_volume']]
    rows.append(dict(stage=job['stage'],cohort=job['cohort'],seed=job['scene']['seed'],inference_seed=job['inference_seed'],first_grasp_step=first,approach_samples=len(approach),negative_intent_samples=len(closes),two_finger_contact_samples=len(both),inside_palm_samples=len(inside),negative_and_contact_samples=sum(r['grasp_after']['planned_apple']['both_fingers_contact'] for r in closes),first_negative_intent=closes[0]['control_step'] if closes else None,first_two_finger_contact=both[0]['control_step'] if both else None,first_inside_palm=inside[0]['control_step'] if inside else None,best_any=best_geometry(approach,lambda r,a:True),best_while_closing=best_geometry(approach,lambda r,a:r['grasp_after']['gripper_intent']<0),best_with_both_fingers=best_geometry(approach,lambda r,a:a['both_fingers_contact']),best_closing_with_both_fingers=best_geometry(approach,lambda r,a:r['grasp_after']['gripper_intent']<0 and a['both_fingers_contact']),replans_before_first_grasp=sum(c['at_control_step']<(first or s['control_steps']+1) for c in chunks),path=str(folder)))
 target=e.ROOT/'analysis'/args.label;target.mkdir(parents=True,exist_ok=True)
 e.save_new(target/'contact_diagnostics.json',dict(rows=rows,annotation='Planned fruit identity and geometry are diagnostic only. Boundary snapshots may miss internal contact events; actual held events remain authoritative. All pregrasp slices exclude boundary at/after first held. Stage labels and observed contact are not policy inputs.',analysis_source_sha256=e.sha(Path(__file__))))
 print(json.dumps(dict(episodes=len(rows),output=str(target/'contact_diagnostics.json'))))
if __name__=='__main__':main()
