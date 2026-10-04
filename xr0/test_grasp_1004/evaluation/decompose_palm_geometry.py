"""CPU-only rigid-pose counterfactual geometry from recorded target/actual traces."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import evaluate_grasp as e

LOWER=np.array([-.025,-.04,.065]);UPPER=np.array([.025,.04,.125])
def geometry(local):
 local=np.asarray(local,dtype=float);margins=np.r_[local-LOWER,UPPER-local]
 return dict(local_xyz_m=local.tolist(),inside_strict=bool(np.all(local>LOWER)&np.all(local<UPPER)),minimum_margin_m=float(margins.min()))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='palm_decomposition');args=p.parse_args()
 out=e.ROOT/'analysis'/args.output;out.mkdir(parents=True,exist_ok=False)
 protocol=json.loads(e.PROTOCOL.read_text());summaries=[];focus={('B30',2014461),('B30',2014362),('B5',2013322),('B5',2014461)}
 for stage in ['B30','B5']:
  for job in protocol['jobs'][stage]:
   folder=e.ROOT/'rollouts'/job['job_id'];steps=e.read_lines(folder/'steps.jsonl');trans=[];rels=[]
   for row in steps:
    rt=Rotation.from_quat(row['measured_quaternion']);snap=row['grasp_after']
    trans.append(rt.inv().apply(np.asarray(snap['hand_position_world'])-row['measured_position']))
    rels.append((rt.inv()*Rotation.from_quat(snap['hand_quaternion_world'])).as_quat())
   trans=np.asarray(trans);translation=np.median(trans,axis=0);relative=Rotation.from_quat(rels[0])
   transform=dict(tcp_to_palm_translation_tcp_frame_m=translation.tolist(),tcp_to_palm_rotation_quaternion=relative.as_quat().tolist(),max_translation_deviation_m=float(np.linalg.norm(trans-translation,axis=1).max()),max_rotation_deviation_rad=float((Rotation.from_quat(rels)*relative.inv()).magnitude().max()))
   assert transform['max_translation_deviation_m']<1e-6 and transform['max_rotation_deviation_rad']<1e-6
   firstheld=next((r['control_step'] for r in steps if r['grasp_after']['held_apple_id'] is not None),None)
   records=[];reconstruction_errors=[]
   for row in steps:
    snap=row['grasp_after'];apple=snap['planned_apple'];fruit=np.asarray(apple['world_position']);pa=np.asarray(row['measured_position']);ra=Rotation.from_quat(row['measured_quaternion']);pt=np.asarray(row['target_position']);rt=Rotation.from_matrix(row['target_rotation'])
    cp=np.asarray(row['info']['target_pose']['position_world']);cr=Rotation.from_quat(row['info']['target_pose']['quaternion_world'])
    def palm_local(pos,rot):return (rot*relative).inv().apply(fruit-(pos+rot.apply(translation)))
    reconstructed=palm_local(pa,ra);reconstruction_errors.append(float(np.linalg.norm(reconstructed-np.asarray(apple['hand_local_position']))))
    actual=geometry(apple['hand_local_position']);target=geometry(palm_local(pt,rt));command=geometry(palm_local(cp,cr))
    record=dict(step=row['control_step'],chunk_id=row['chunk_id'],target_k=row['target_k'],dwell_steps=row['dwell_steps'],reached=row['reached'],advance=row['advance'],advance_reason=row['advance_reason'],first_grasp_step=firstheld,pregrasp=firstheld is None or row['control_step']<firstheld,actual=actual,predicted_target=target,attempted_clipped_command=command,target_translation_actual_rotation=geometry(palm_local(pt,ra)),actual_translation_target_rotation=geometry(palm_local(pa,rt)),tracking_position_m=float(np.linalg.norm(pt-pa)),tracking_rotation_rad=float((rt*ra.inv()).magnitude()),model_to_attempted_command_position_m=float(np.linalg.norm(pt-cp)),attempted_command_to_actual_position_m=float(np.linalg.norm(cp-pa)),model_to_attempted_command_rotation_rad=float((rt*cr.inv()).magnitude()),attempted_command_to_actual_rotation_rad=float((cr*ra.inv()).magnitude()),ik_failed=row['info']['ik_failed'],action_clipped=row['info']['action_clipped'],negative_intent=snap['gripper_intent']<0,both_finger_contact=apple['both_fingers_contact'],target_width_m=row['target_width'],commanded_width_m=snap['commanded_width_m'],actual_width_m=row['measured_width'],held_apple_id=snap['held_apple_id'])
    records.append(record)
   transform['max_actual_local_reconstruction_error_m']=max(reconstruction_errors);assert max(reconstruction_errors)<1e-6
   filtered=[r for r in records if r['pregrasp']]
   masks={'all_pregrasp':lambda r:True,'negative_intent':lambda r:r['negative_intent'],'both_contact':lambda r:r['both_finger_contact'],'negative_and_both':lambda r:r['negative_intent'] and r['both_finger_contact'],'negative_both_reached':lambda r:r['negative_intent'] and r['both_finger_contact'] and r['reached']}
   mask_stats={}
   for name,fn in masks.items():
    rr=[r for r in filtered if fn(r)]
    mask_stats[name]=dict(samples=len(rr),actual_inside=sum(r['actual']['inside_strict'] for r in rr),target_inside=sum(r['predicted_target']['inside_strict'] for r in rr),command_inside=sum(r['attempted_clipped_command']['inside_strict'] for r in rr),target_inside_actual_outside=sum(r['predicted_target']['inside_strict'] and not r['actual']['inside_strict'] for r in rr),command_inside_actual_outside=sum(r['attempted_clipped_command']['inside_strict'] and not r['actual']['inside_strict'] for r in rr),target_outside_actual_outside=sum(not r['predicted_target']['inside_strict'] and not r['actual']['inside_strict'] for r in rr),best_actual=max(rr,key=lambda r:r['actual']['minimum_margin_m']) if rr else None,best_target=max(rr,key=lambda r:r['predicted_target']['minimum_margin_m']) if rr else None,first_target_inside_actual_outside=next((r for r in rr if r['predicted_target']['inside_strict'] and not r['actual']['inside_strict']),None))
   summary=dict(stage=stage,seed=job['scene']['seed'],transform=transform,masks=mask_stats,input_steps_sha256=e.sha(folder/'steps.jsonl'),path=str(folder))
   if (stage,job['scene']['seed']) in focus:
    detail=out/f'{stage}_{job["scene"]["seed"]}_steps.jsonl'
    with detail.open('x') as f:
     for row in records:f.write(json.dumps(row,allow_nan=False)+'\n')
    summary['detailed_trace']=str(detail)
   summaries.append(summary)
 result=dict(protocol_sha256=e.sha(e.PROTOCOL),analysis_source_sha256=e.sha(Path(__file__)),cpu_only=True,simulation_rerun=False,source_logs_modified=False,method='Infer near-fixed TCP-to-palm rigid transform from measured poses. Hold the apple at its observed post-step world position, substitute predicted TCP target (or attempted clipped target), and transform that observed apple into the counterfactual palm. Width/contact are observed separately; no counterfactual contact or grasp is asserted.',limitations=['This is a geometric counterfactual at the observed apple position, not a physics rerun. Moving the palm could change fruit motion and finger contacts.','attempted_clipped_command is the logged controller target after action/workspace clipping; IK failure may reject it.','Target inside plus actual outside supports a tracking contribution at that boundary, but does not prove waiting longer would establish grasp.','Actual reach thresholds are broader than some remaining palm margins; reached=true is not a grasp geometry guarantee.'],episodes=summaries)
 e.save_new(out/'geometry_decomposition.json',result)
 print(json.dumps(dict(output=str(out),episodes=len(summaries),rigid_transform_max_deviation_m=max(r['transform']['max_translation_deviation_m'] for r in summaries)),indent=2))
 for r in summaries:
  if (r['stage'],r['seed']) in focus:
   print(r['stage'],r['seed'],'negative_both counts',{k:v for k,v in r['masks']['negative_and_both'].items() if not isinstance(v,dict)})
if __name__=='__main__':main()
