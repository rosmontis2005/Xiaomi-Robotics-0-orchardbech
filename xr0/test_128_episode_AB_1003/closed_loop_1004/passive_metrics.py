"""Postprocessing and read-only diagnostic fields; never used to choose actions."""
import numpy as np
from scipy.spatial.transform import Rotation


def planned_apple_observation(snapshot,apple_id,body,world_position):
 world=np.asarray(world_position,dtype=float);hand=np.asarray(snapshot['hand_position_world'])
 local=Rotation.from_quat(snapshot['hand_quaternion_world']).inv().apply(world-hand)
 contacts={a['apple_id']:a for a in snapshot['contact_apples']};contact=contacts.get(apple_id,{})
 touching=contact.get('touching_finger_bodies',[])
 both=set(touching)==set(snapshot['expected_finger_bodies']) and len(touching)==2
 inside=abs(local[0])<.025 and abs(local[1])<.04 and .065<local[2]<.125
 lower=np.array([-.025,-.04,.065]);upper=np.array([.025,.04,.125])
 return dict(apple_id=int(apple_id),apple_body=int(body),world_position=world.tolist(),hand_local_position=local.tolist(),
   tcp_distance_m=float(np.linalg.norm(world-np.asarray(snapshot['tcp_position_world']))),
   palm_center_distance_m=float(np.linalg.norm(local-(lower+upper)/2)),
   palm_volume_distance_m=float(np.linalg.norm(np.maximum(lower-local,0)+np.maximum(local-upper,0))),
   inside_palm_volume=bool(inside),touching_finger_bodies=touching,both_fingers_contact=both,geometry_eligible=bool(inside and both),
   diagnostic_only=True)


def distribution(values):
 a=np.asarray(list(values),dtype=float)
 return None if not a.size else dict(count=int(a.size),mean=float(a.mean()),p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)),max=float(a.max()))


def stamp(row,boundary='after',apple=None):
 snap=row['grasp_'+boundary]
 return dict(control_step=row['control_step'],boundary=boundary,completed_control_steps=row['control_step']-(boundary=='before'),chunk_id=row['chunk_id'],target_k=row['target_k'],target_width_m=row['target_width'],commanded_width_at_boundary_m=snap['commanded_width_m'],actual_width_m=snap['measured_width_m'],intent=snap['gripper_intent'],held_apple_id=snap['held_apple_id'],apple=apple)


def diagnostics(steps,model_chunks,gt_chunks):
 first=next((r['control_step'] for r in steps if r['info']['grasp_assist_triggered'] or r['grasp_after']['held_apple_id'] is not None),None)
 candidates=[];nearest=[]
 for row in steps:
  for boundary in ['before','after']:
   complete=row['control_step']-(boundary=='before')
   if first is not None and complete>=first:continue
   snap=row['grasp_'+boundary]
   if snap.get('planned_apple') is not None:candidates.append((snap['planned_apple']['tcp_distance_m'],stamp(row,boundary,snap['planned_apple'])))
   if snap.get('nearest_tcp_apple') is not None:nearest.append((snap['nearest_tcp_apple']['tcp_distance_m'],stamp(row,boundary,snap['nearest_tcp_apple'])))
 releases=[stamp(r) for r in steps if r['grasp_before']['held_apple_id'] is not None and r['grasp_after']['held_apple_id'] is None]
 first_ik=next((stamp(r) for r in steps if r['info']['ik_failed']),None)
 after_ik=next((stamp(r) for r in steps if first is not None and r['control_step']>first and r['info']['ik_failed']),None)
 first_close=next((stamp(r) for r in steps if r['grasp_after']['gripper_intent']<0),None)
 errors=None
 if model_chunks and gt_chunks:
  m,g=model_chunks[0],gt_chunks[0]
  p=np.linalg.norm(np.asarray(m['target_positions'])-np.asarray(g['target_positions']),axis=-1)
  rm=np.asarray(m['target_rotations']);rg=np.asarray(g['target_rotations'])
  rot=np.arccos(np.clip((np.einsum('tij,tij->t',rm,rg)-1)/2,-1,1))
  widths=np.abs(np.asarray(m['target_widths'])-np.asarray(g['target_widths']))
  assert len(p)==len(rot)==len(widths)==30
  errors={label:dict(position_error_m=distribution(p[:n]),rotation_error_rad=distribution(rot[:n]),width_absolute_error_m=distribution(widths[:n])) for label,n in [('first5',5),('full30',30)]}
  errors['interpretation']='Unclipped decoded first-chunk world targets versus teacher targets at the same reset; fitting diagnostic only, not a requirement to choose the teacher fruit.'
 return dict(minimum_tcp_to_planned_apple_before_first_grasp=min(candidates,key=lambda x:x[0])[1] if candidates else None,
   minimum_tcp_to_nearest_apple_before_first_grasp=min(nearest,key=lambda x:x[0])[1] if nearest else None,
   planned_fruit_samples_available=len(candidates),first_chunk_teacher_error=errors,
   first_ik_failure=first_ik,first_ik_after_first_grasp=after_ik,
   observed_held_to_none_release_count=len(releases),observed_held_to_none_release_events=releases,
   first_negative_intent_boundary=first_close,
   interpretation='Boundary snapshots exclude all samples at/after first grasp from approach-distance statistics. Held-to-none is observed release, not an assertion about every internal physics substep. Planned fruit identity is evaluator-only and never a grasp success gate.')
