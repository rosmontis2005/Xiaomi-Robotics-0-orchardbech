#!/usr/bin/env python3
"""CPU-only same-fruit boundary evidence classification; reads completed jobs only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from scipy.spatial.transform import Rotation
import argparse,ast,collections,datetime,hashlib,json,math
import numpy as np
ROOT=Path(__file__).resolve().parent;EXPERIMENT=ROOT.parent;EVAL=EXPERIMENT/'evaluation';ORCHARD=Path('/home/rosmontis/Projects/orchardbench')

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def save(p,x):
 p=Path(p).resolve();assert p.is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def bucket_constants():
 want={'_CHASSIS','_CHASSIS_Z','_BUCKET_CENTER_X','_BUCKET_HALF','_BUCKET_WALL_H'};out={}
 for n in ast.parse((ORCHARD/'treesim/robot.py').read_text()).body:
  if isinstance(n,ast.Assign):
   for t in n.targets:
    if isinstance(t,ast.Name) and t.id in want:out[t.id]=ast.literal_eval(n.value)
 assert set(out)==want;return out
BUCKET=bucket_constants()
def bucket_geometry(world,base,yaw):
 p=np.asarray(world)-np.asarray(base);c,s=math.cos(yaw),math.sin(yaw);x=c*p[0]+s*p[1];y=-s*p[0]+c*p[1];z=p[2];floor=BUCKET['_CHASSIS_Z']+BUCKET['_CHASSIS'][2];half=BUCKET['_BUCKET_HALF'];cx=BUCKET['_BUCKET_CENTER_X'];height=BUCKET['_BUCKET_WALL_H'];viol=np.array([max(abs(x-cx)-half,0),max(abs(y)-half,0),max(floor-z,z-(floor+height),0)])
 return dict(local_position_m=[float(x),float(y),float(z)],inside_geometry=bool(abs(x-cx)<half and abs(y)<half and floor<z<floor+height),distance_to_bucket_volume_m=float(np.linalg.norm(viol)),horizontal_distance_to_bucket_center_m=float(np.linalg.norm([x-cx,y])),note='Approximate diagnostic using recorded initial chassis position/yaw; small weld residual/roll/pitch is not captured. Authoritative occupancy is recorded same-fruit in_bucket ID.')
def apple_in_snapshot(snap,fruit):
 for name in ['planned_apple','nearest_tcp_apple','nearest_palm_center_apple']:
  a=snap.get(name)
  if a is not None and a['apple_id']==fruit:return a
 return next((a for a in snap.get('contact_apples',[]) if a['apple_id']==fruit),None)
def margin(local):
 x,y,z=map(float,local);margins=dict(x=.025-abs(x),y=.04-abs(y),z_lower=z-.065,z_upper=.125-z)
 return dict(min_margin_m=min(margins.values()),axis_margins_m=margins,violated_axes=[k for k,v in margins.items() if v<=0])
def evidence(r,boundary='after',fruit=None,base=None,yaw=None):
 snap=r['grasp_'+boundary];a=snap['planned_apple'] if fruit is None else apple_in_snapshot(snap,fruit);out=dict(control_step=r['control_step'],boundary=boundary,completed_control_steps=r['control_step']-(boundary=='before'),sim_time_s=r['sim_time']-(1/30 if boundary=='before' else 0),chunk_id=r['chunk_id'],target_k=r['target_k'],gt_target_index=r.get('gt_target_index'),gt_phase=r.get('gt_phase'),command_target_width_m=r['target_width'],boundary_commanded_width_m=snap['commanded_width_m'],boundary_measured_width_m=snap['measured_width_m'],boundary_gripper_intent=snap['gripper_intent'],boundary_held_apple_id=snap['held_apple_id'],intent_allows_attach=snap['intent_allows_attach'],eligible_apple_ids=snap['eligible_apple_ids'],ambiguous_eligible_apples=snap['ambiguous_eligible_apples'],tracking_position_error_after_m=r['position_error_m'],tracking_rotation_error_after_rad=r['rotation_error_rad'],ik_failed_during_step=r['info']['ik_failed'],advance_reason=r['advance_reason'],dwell_steps=r['dwell_steps'])
 if a is not None:
  out['fruit']=dict(apple_id=a['apple_id'],world_position=a['world_position'],hand_local_position=a['hand_local_position'],tcp_distance_m=a['tcp_distance_m'],inside_palm_volume=a['inside_palm_volume'],both_fingers_contact=a['both_fingers_contact'],geometry_eligible=a['geometry_eligible'],**margin(a['hand_local_position']))
  if base is not None:out['fruit']['bucket_geometry']=bucket_geometry(a['world_position'],base,yaw)
 else:out['fruit']=None
 return out

def front_evidence(steps,first_grasp):
 observations=[]
 for r in steps:
  if r['control_step']>300:break
  for boundary in ['before','after']:
   snap=r['grasp_'+boundary]
   if snap['held_apple_id'] is not None:continue
   if first_grasp is not None and r['control_step']>first_grasp:continue
   if first_grasp is not None and r['control_step']==first_grasp and boundary=='after':continue
   observations.append(evidence(r,boundary))
 def select(predicate,score=None):
  valid=[e for e in observations if predicate(e)];return None if not valid else(min(valid,key=score) if score else valid[0])
 closest=select(lambda e:e['fruit'] is not None,lambda e:e['fruit']['tcp_distance_m']);best=select(lambda e:e['fruit'] is not None,lambda e:-e['fruit']['min_margin_m']);bothclose=select(lambda e:e['fruit'] is not None and e['fruit']['both_fingers_contact'] and e['boundary_gripper_intent']<0,lambda e:-e['fruit']['min_margin_m']);goodnonclosing=select(lambda e:e['fruit'] is not None and e['fruit']['inside_palm_volume'] and e['fruit']['both_fingers_contact'] and not e['intent_allows_attach']);goodconditions=select(lambda e:e['fruit'] is not None and e['fruit']['inside_palm_volume'] and e['fruit']['both_fingers_contact'] and e['intent_allows_attach'] and len(e['eligible_apple_ids'])==1);insideclosing=select(lambda e:e['fruit'] is not None and e['fruit']['inside_palm_volume'] and e['intent_allows_attach'] and not e['fruit']['both_fingers_contact']);firstclosing=select(lambda e:e['boundary_gripper_intent']<0)
 counts=dict(boundary_observations=len(observations),inside_palm=sum(e['fruit'] is not None and e['fruit']['inside_palm_volume'] for e in observations),dual_contact=sum(e['fruit'] is not None and e['fruit']['both_fingers_contact'] for e in observations),closing_intent=sum(e['boundary_gripper_intent']<0 for e in observations),inside_plus_dual_plus_closing=sum(e['fruit'] is not None and e['fruit']['inside_palm_volume'] and e['fruit']['both_fingers_contact'] and e['intent_allows_attach'] for e in observations))
 if first_grasp is not None and first_grasp<=300:label='GRASP_OBSERVED_BY300'
 elif goodconditions is not None:label='BOUNDARY_CONDITIONS_PRESENT_WITHOUT_OBSERVED_GRASP'
 elif goodnonclosing is not None:label='PALM_AND_DUAL_CONTACT_WITHOUT_CLOSING'
 elif bothclose is not None and bothclose['fruit']['min_margin_m']<=0:label='CLOSING_DUAL_CONTACT_OUTSIDE_PALM'
 elif insideclosing is not None:label='INSIDE_PALM_CLOSING_WITHOUT_DUAL_CONTACT'
 elif counts['inside_palm']==0:label='PLANNED_FRUIT_NEVER_ENTERED_PALM'
 else:label='INSIDE_PALM_WITH_INCOMPLETE_CONTACT_INTENT_EVIDENCE'
 return dict(classification=label,any_fruit_grasped_by300=first_grasp is not None and first_grasp<=300,counts=counts,nearest_planned_fruit=closest,best_planned_palm_margin=best,best_closing_dual_contact_margin=bothclose,first_closing_intent=firstclosing,first_palm_dual_nonclosing=goodnonclosing,first_all_boundary_conditions=goodconditions,first_inside_closing_missing_dual=insideclosing,note='Observations stop before first actual grasp of ANY fruit. Planned-fruit diagnostic is separate from actual held fruit. Before/after boundary samples are adjacent repeated observations, not independent trials or unique causal attribution.')

def reconstruct_fruits(steps):
 chains={};previous=None;run=0;detached_first={};releases=[];success=[];previousdet=set()
 for r in steps:
  step=r['control_step'];fs=r['fruit_state'];held=fs['held_apple_id'];det=set(fs['detached_apple_ids']);bucket=set(fs['in_bucket_apple_ids'])
  assert held==r['grasp_after']['held_apple_id']
  for fruit in det-previousdet:detached_first.setdefault(fruit,step)
  previousdet=det
  if held is not None:
   c=chains.setdefault(held,dict(first_grasp_step=step,held15_step=None,first_detach_after_grasp_step=None,release_step=None,stable_bucket_steps=0,max_stable_bucket_steps=0,strict_success_step=None,max_held_steps=0,held_in_bucket_first_step=None,held_in_bucket_boundaries=0,regrasp_steps=[]))
   if previous!=held and step!=c['first_grasp_step']:c['regrasp_steps'].append(step)
   run=run+1 if held==previous else 1;c['max_held_steps']=max(c['max_held_steps'],run)
   if run>=15 and c['held15_step'] is None:c['held15_step']=step
   c['release_step']=None
   if held in bucket:
    if c['held_in_bucket_first_step'] is None:c['held_in_bucket_first_step']=step
    c['held_in_bucket_boundaries']+=1
  else:run=0
  for fruit,c in chains.items():
   d=detached_first.get(fruit)
   if d is not None and d>=c['first_grasp_step'] and c['first_detach_after_grasp_step'] is None:c['first_detach_after_grasp_step']=d
  if previous is not None and previous!=held:
   c=chains[previous];valid=c['held15_step'] is not None and previous in det and c['first_detach_after_grasp_step'] is not None;c['release_step']=step if valid else None;releases.append(dict(apple_id=previous,step=step,had_held15=c['held15_step'] is not None,detached_at_release=previous in det,valid_chain=valid,in_bucket_at_release=previous in bucket,replacement_held_apple_id=held))
  for fruit,c in chains.items():
   stable=c['release_step'] is not None and held!=fruit and fruit in det and fruit in bucket;c['stable_bucket_steps']=c['stable_bucket_steps']+1 if stable else 0;c['max_stable_bucket_steps']=max(c['max_stable_bucket_steps'],c['stable_bucket_steps'])
   if c['stable_bucket_steps']>=60 and c['strict_success_step'] is None:c['strict_success_step']=step;success.append(dict(apple_id=fruit,step=step,release_step=c['release_step'],held15_step=c['held15_step'],first_detach_after_grasp_step=c['first_detach_after_grasp_step']))
  assert bool(success)==r['strict_success_at_boundary'];previous=held
 return chains,releases,success

def short_commit(steps,chunks):
 firstclose=next((r['control_step'] for r in steps if r['grasp_after']['gripper_intent']<0),None)
 rows=[]
 for c in chunks:
  rr=[r for r in steps if r['chunk_id']==c['chunk_id']]
  if not rr:continue
  anchor=np.asarray(c['anchor']['tcp_pos_world']);rotation=Rotation.from_quat(c['anchor']['tcp_quat_world']).as_matrix();goals=np.asarray(c['target_positions']);rotations=np.asarray(c['target_rotations']);relative=np.linalg.norm(goals-anchor,axis=1);angles=Rotation.from_matrix(rotation.T@rotations).magnitude();first=rr[0];last=rr[-1];startfruit=first['grasp_before']['planned_apple'];endfruit=last['grasp_after']['planned_apple'];seen=sorted({r['target_k'] for r in rr})
  item=dict(chunk_id=c['chunk_id'],replan_at_control_step=c['at_control_step'],processed_target_limit=c['stop_k']-c['start_k'],control_steps=len(rr),first5_goal_position_delta_from_anchor_m=relative[:5].tolist(),first5_goal_rotation_delta_from_anchor_rad=angles[:5].tolist(),first5_goal_position_le10mm_count=int((relative[:5]<=.01).sum()),first5_goal_position_le10mm_and_rotation_le0_08rad_count=int(((relative[:5]<=.01)&(angles[:5]<=.08)).sum()),first5_endpoint_position_delta_m=float(relative[min(4,len(relative)-1)]),full30_endpoint_position_delta_m=float(relative[-1]),first5_goal_widths_m=c['target_widths'][:5],actual_TCP_net_displacement_m=float(np.linalg.norm(np.asarray(last['measured_position'])-anchor)),actual_TCP_net_vector_m=(np.asarray(last['measured_position'])-anchor).tolist(),start_planned_tcp_distance_m=startfruit['tcp_distance_m'],end_planned_tcp_distance_m=endfruit['tcp_distance_m'],start_planned_palm_xyz_m=startfruit['hand_local_position'],end_planned_palm_xyz_m=endfruit['hand_local_position'],planned_distance_reduction_m=startfruit['tcp_distance_m']-endfruit['tcp_distance_m'],executed_target_indices=seen,executed_unique_goal_position_le10mm_count=int((relative[seen]<=.01).sum()),reached_advances=sum(r['advance_reason']=='reached' for r in rr),dwell_timeout_advances=sum(r['advance_reason']=='maximum_dwell' for r in rr),first_actual_negative_intent_step=next((r['control_step'] for r in rr if r['grasp_after']['gripper_intent']<0),None),first5_prediction_note='Threshold is descriptive relative to inference anchor; executing target tolerance is evaluated against live TCP and rotation. Small prefix does not itself establish failure cause.')
  rows.append(item)
 sections={}
 for name,end in [('first300',min(300,len(steps))),('before_first_close',min(300,len(steps),firstclose-1) if firstclose is not None else min(300,len(steps)))]:
  selected=[]
  for c in chunks:
   rr=[r for r in steps if r['chunk_id']==c['chunk_id'] and r['control_step']<=end]
   if not rr:continue
   anchor=np.asarray(c['anchor']['tcp_pos_world']);seen=sorted({r['target_k'] for r in rr});goals=np.asarray(c['target_positions']);relative=np.linalg.norm(goals-anchor,axis=1);a=rr[0]['grasp_before']['planned_apple'];z=rr[-1]['grasp_after']['planned_apple'];selected.append(dict(chunk_id=c['chunk_id'],replan_at_control_step=c['at_control_step'],partial_at_cutoff=rr[-1]['control_step']<max(r['control_step'] for r in steps if r['chunk_id']==c['chunk_id']),observed_control_steps=len(rr),observed_unique_targets=len(seen),goal_position_le10mm_count=int((relative[seen]<=.01).sum()),actual_TCP_net_displacement_m=float(np.linalg.norm(np.asarray(rr[-1]['measured_position'])-anchor)),start_planned_tcp_distance_m=a['tcp_distance_m'],end_planned_tcp_distance_m=z['tcp_distance_m'],planned_distance_reduction_m=a['tcp_distance_m']-z['tcp_distance_m'],start_planned_palm_xyz_m=a['hand_local_position'],end_planned_palm_xyz_m=z['hand_local_position'],reached_advances=sum(r['advance_reason']=='reached' for r in rr),dwell_timeout_advances=sum(r['advance_reason']=='maximum_dwell' for r in rr),mean_tracking_position_error_after_m=float(np.mean([r['position_error_m'] for r in rr])),mean_tracking_rotation_error_after_rad=float(np.mean([r['rotation_error_rad'] for r in rr]))))
  total=sum(r['observed_unique_targets'] for r in selected);sections[name]=dict(control_step_cutoff=end,chunks=len(selected),unique_chunk_target_pairs=total,goal_position_le10mm_fraction=None if not total else sum(r['goal_position_le10mm_count'] for r in selected)/total,chunk_actual_net_displacement_mean_m=None if not selected else float(np.mean([r['actual_TCP_net_displacement_m'] for r in selected])),sum_chunk_actual_net_displacement_m=sum(r['actual_TCP_net_displacement_m'] for r in selected),planned_distance_reduction_sum_m=sum(r['planned_distance_reduction_m'] for r in selected),reached_advances=sum(r['reached_advances'] for r in selected),dwell_timeout_advances=sum(r['dwell_timeout_advances'] for r in selected),rows=selected)
 return dict(first_actual_negative_intent_step=firstclose,sections=sections,chunks=rows,interpretation='Evidence for testing repeated short-prefix re-anchoring / delayed approach / early closing; descriptive only. Position10mm is not an asserted root cause. Rotational errors, live reached events and dwell timeouts remain separate.')

def classify_episode(folder,job,protocol_sha):
 done=load(folder/'completed.json');assert done['status']=='COMPLETE' and done['job_id']==job['job_id'] and done['protocol_sha256']==protocol_sha
 for name,digest in done['files_sha256'].items():assert sha(folder/name)==digest
 summary=load(folder/'summary.json');metrics=load(folder/'metrics.json');steps=lines(folder/'steps.jsonl');chunks=lines(folder/'chunks.jsonl');initial=load(folder/'initial.json');assert len(steps)==summary['control_steps'] and [r['control_step'] for r in steps]==list(range(1,len(steps)+1));assert metrics['valid_for_outcome'];assert all(all(r['observer_consistency'].values()) for r in steps)
 base=initial['obs']['base_pos'];yaw=initial['obs']['base_yaw'];baseanchors=[c['anchor'] for c in chunks if 'base_pos' in c['anchor']];base_position_drift=max((float(np.linalg.norm(np.asarray(c['base_pos'])-base)) for c in baseanchors),default=None);base_yaw_drift=max((abs(c['base_yaw']-yaw) for c in baseanchors),default=None);assert baseanchors or job['is_gt']
 planned={r['grasp_after']['planned_apple']['apple_id'] for r in steps};assert len(planned)==1;planned_id=next(iter(planned));chains,releases,success=reconstruct_fruits(steps);saved=summary['strict_placement'];assert bool(success)==summary['strict_success']==metrics['strict_success'] and success==saved['strict_success_events'];assert set(map(str,chains))==set(saved['fruit_chains'])
 for fruit,c in chains.items():
  for k,v in saved['fruit_chains'][str(fruit)].items():assert c[k]==v,(fruit,k,c[k],v)
 for a,b in zip(releases,saved['release_events']):assert all(a[k]==v for k,v in b.items())
 assert len(releases)==len(saved['release_events'])
 bucket_matches=0;bucket_approx_mismatch=0
 for r in steps:
  a=r['grasp_after']['planned_apple'];pred=bucket_geometry(a['world_position'],base,yaw)['inside_geometry'] and planned_id in r['fruit_state']['detached_apple_ids'];actual=planned_id in r['fruit_state']['in_bucket_apple_ids'];bucket_matches+=int(pred==actual);bucket_approx_mismatch+=int(pred!=actual)
 for release in releases:
  r=steps[release['step']-1];fruit=release['apple_id'];ev=evidence(r,'after',fruit,base,yaw)
  if ev['fruit'] is None:ev=evidence(r,'before',fruit,base,yaw)
  release['evidence']=ev;geo=None if ev['fruit'] is None else ev['fruit']['bucket_geometry'];release['distance_to_bucket_volume_m']=None if geo is None else geo['distance_to_bucket_volume_m'];release['outside_bucket_at_release']=not release['in_bucket_at_release'];release['far_release_diagnostic_gt0_10m']=geo is not None and geo['distance_to_bucket_volume_m']>.10
  later=[r2 for r2 in steps if r2['control_step']>=release['step']];first_regrasp=next((r2['control_step'] for r2 in later if r2['fruit_state']['held_apple_id']==fruit),None);episode_end=(first_regrasp-1) if first_regrasp is not None else len(steps);release['observation_end_before_regrasp_step']=episode_end;release['remaining_boundaries_before_end_or_regrasp']=episode_end-release['step']+1;release['first_bucket_step_after_release_before_regrasp']=next((r2['control_step'] for r2 in later if r2['control_step']<=episode_end and fruit in r2['fruit_state']['in_bucket_apple_ids']),None)
 fruit_classes=[]
 for fruit,c in chains.items():
  rs=[r for r in releases if r['apple_id']==fruit];valid=[r for r in rs if r['valid_chain']];finalheld=steps[-1]['fruit_state']['held_apple_id']==fruit
  if c['strict_success_step'] is not None:label='STRICT_RELEASE_BUCKET60'
  elif c['held15_step'] is None:label='TRANSIENT_GRASP_LT15'
  elif not rs:label='IN_BUCKET_STILL_HELD' if c['held_in_bucket_first_step'] is not None else 'HELD_NO_RELEASE'
  elif finalheld:label='REGRASPED_STILL_HELD_AFTER_RELEASE'
  elif not valid:label='RELEASE_WITHOUT_VALID_SAMEFRUIT_DETACH_CHAIN'
  elif any(r['in_bucket_at_release'] for r in valid):label='RELEASED_IN_BUCKET_BUT_NO60_STABILITY'
  elif any(r['first_bucket_step_after_release_before_regrasp'] is not None for r in valid):label='RELEASED_OUTSIDE_THEN_BUCKET_BUT_NO60_STABILITY'
  elif all(r['far_release_diagnostic_gt0_10m'] for r in valid):label='RELEASED_FAR_FROM_BUCKET'
  else:label='RELEASED_OUTSIDE_BUCKET'
  firstrow=steps[c['first_grasp_step']-1];post=[r for r in steps if r['control_step']>=c['first_grasp_step']];fruit_classes.append(dict(apple_id=fruit,is_planned_fruit=fruit==planned_id,classification=label,chain=c,first_grasp_evidence=evidence(firstrow,'after',fruit,base,yaw),releases=rs,first_postgrasp_ik_step=next((r['control_step'] for r in post if r['info']['ik_failed']),None),postgrasp_ik_failed_steps=sum(r['info']['ik_failed'] for r in post),postgrasp_dwell_timeouts=sum(r['advance_reason']=='maximum_dwell' for r in post)))
 first=min((c['first_grasp_step'] for c in chains.values()),default=None);held15=min((c['held15_step'] for c in chains.values() if c['held15_step'] is not None),default=None);front=front_evidence(steps,first)
 if success:outcome='STRICT_RELEASE_BUCKET60'
 elif not fruit_classes:outcome='NO_GRASP'
 elif len(fruit_classes)==1:outcome=fruit_classes[0]['classification']
 else:outcome='MULTIPLE_FRUITS_NO_STRICT_SUCCESS'
 pre=[r for r in steps if r['control_step']<=300 and(first is None or r['control_step']<first)];firstlegacy=next((r['control_step'] for r in steps if r['info']['success']),None)
 return dict(job_id=job['job_id'],stage=job['stage'],cohort=job['cohort'],scene_seed=job['scene']['seed'],episode_id=job['scene']['episode_id'],split=job['scene']['split'],inference_seed=job['inference_seed'],processed_targets=job['processed_targets'],is_gt=job['is_gt'],status='COMPLETE',outcome=outcome,planned_apple_id=planned_id,actual_grasped_apple_ids=sorted(chains),first_grasp_step=first,first_held15_step=held15,held15=held15 is not None,held15_by300=held15 is not None and held15<=300,any_grasp_by300=first is not None and first<=300,strict_success=bool(success),strict_events=success,legacy_success_ever=firstlegacy is not None,first_legacy_success_step=firstlegacy,held_fruit_at_first_legacy_success=None if firstlegacy is None else steps[firstlegacy-1]['fruit_state']['held_apple_id'],final_held_apple_id=steps[-1]['fruit_state']['held_apple_id'],front300=front,fruits=fruit_classes,first_ik_failure_step=next((r['control_step'] for r in steps if r['info']['ik_failed']),None),pre300_pregrasp_ik_failed_steps=sum(r['info']['ik_failed'] for r in pre),pre300_pregrasp_dwell_timeouts=sum(r['advance_reason']=='maximum_dwell' for r in pre),control_steps=len(steps),replans=summary['replans'],branch_break_count=summary['max_branch_break_count'],first_chunk_teacher_error=metrics.get('first_chunk_teacher_error'),short_commit_diagnostics=None if job['is_gt'] else short_commit(steps,chunks),pregrasp_replan_control_steps=[c['at_control_step'] for c in chunks if first is None or c['at_control_step']<first],path=str(folder),validation=dict(completion_files_hashes=True,strict_samefruit_reconstruction_matches=True,all_boundary_observers_consistent=True,checked_base_chunk_boundaries=len(baseanchors),base_position_drift_from_initial_m=base_position_drift,base_yaw_drift_from_initial_rad=base_yaw_drift,approx_bucket_geometry_membership_matches_steps=bucket_matches,approx_bucket_geometry_mismatches=bucket_approx_mismatch),completion_sha256=sha(folder/'completed.json'))

def collect(protocol_dirs=None):
 protocol_dirs=protocol_dirs or [EVAL,EVAL/'L8000'];rows=[];pending=[];groups={};source=[]
 for root in map(Path,protocol_dirs):
  path=root/'protocol.json'
  if not path.exists():pending.append(dict(protocol=str(path),reason='PROTOCOL_NOT_PRESENT'));continue
  protocol=load(path);digest=sha(path);source.append(dict(path=str(path),sha256=digest))
  for stage,jobs in protocol['jobs'].items():
   current=[]
   for job in jobs:
    folder=root/'rollouts'/job['job_id']
    if not(folder/'completed.json').is_file():pending.append(dict(stage=stage,job_id=job['job_id'],directory_exists=folder.exists(),reason='NO_COMPLETED_MARKER_NOT_ANALYZED'));continue
    row=classify_episode(folder,job,digest);rows.append(row);current.append(row)
   complete=len(current)==len(jobs);groups[stage]=dict(status='COMPLETE' if complete else 'PARTIAL',completed=len(current),expected=len(jobs),counts_are_final=complete,outcomes=dict(collections.Counter(r['outcome'] for r in current)),front300_classes=dict(collections.Counter(r['front300']['classification'] for r in current)),held15=sum(r['held15'] for r in current),held15_by300=sum(r['held15_by300'] for r in current),strict_success=sum(r['strict_success'] for r in current),legacy_success_ever=sum(r['legacy_success_ever'] for r in current))
 return dict(schema='orchard_boundary_failure_evidence_v1',status='COMPLETE' if not pending else 'PARTIAL',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_protocols=source,analysis_source_sha256=sha(__file__),groups=groups,rows=rows,pending=pending,limitations=['Only completed+hash-valid episodes enter counts. Incomplete stages are explicitly PARTIAL and are not final results.','Front300 boundaries are compatible evidence, not proof of unique causal mechanism or all substep contact states.','Planned fruit diagnostics and actual held fruit chains are separate; grasp of another fruit counts as actual grasp.','Same-fruit detach may precede held15 but must occur at or after first grasp and by release.','Far-release >0.10m is an explanatory diagnostic, not a new success criterion.','Bucket60/held15 are consecutive control-boundary samples; no claim about unobserved physics substeps.','No fresh24 heldout model data is read; no simulator or GPU imports.'])

def self_test():
 # Distinct fruit identity must never satisfy another fruit's release/bucket chain.
 def fake(held,det,bucket,i):return dict(control_step=i,fruit_state=dict(held_apple_id=held,detached_apple_ids=det,in_bucket_apple_ids=bucket),grasp_after=dict(held_apple_id=held),strict_success_at_boundary=False)
 rows=[fake(1,[] if i<5 else[1],[],i) for i in range(1,16)]+[fake(None,[1],[2],i)for i in range(16,76)];c,r,s=reconstruct_fruits(rows);assert not s and r[0]['valid_chain'] and c[1]['max_stable_bucket_steps']==0 and c[1]['first_detach_after_grasp_step']==5
 rows=[fake(1,[] if i<5 else[1],[],i)for i in range(1,16)]+[fake(None,[1],[1],i)for i in range(16,76)];rows[-1]['strict_success_at_boundary']=True;c,r,s=reconstruct_fruits(rows);assert s[0]['step']==75 and c[1]['held15_step']==15 and c[1]['first_detach_after_grasp_step']==5
 assert margin([.027,.0,.10])['violated_axes']==['x'] and abs(margin([.027,0,.1])['min_margin_m']+.002)<1e-12
 result=collect([EVAL]);assert result['groups']['B30']['status']=='COMPLETE' and result['groups']['B30']['completed']==8 and result['groups']['B30']['held15']==6 and result['groups']['B30']['strict_success']==0
 return dict(status='PASS',same_fruit_strict_sequence_verified=True,detach_before_held15_allowed=True,wrong_fruit_bucket_rejected=True,palm_2mm_violation_exact=True,real_B30_completed8=8,real_B30_held15=6,real_B30_strict=0,real_completed_episodes_checked=len(result['rows']),partial_jobs_excluded=len(result['pending']),no_torch_or_simulator_imports=not any(x in sys.modules for x in ['torch','treesim.vla_env','warp']))

def main():
 p=argparse.ArgumentParser();p.add_argument('--label',default='closed_loop_snapshot');p.add_argument('--preflight',action='store_true');p.add_argument('--only-d0',action='store_true');args=p.parse_args()
 if args.preflight:r=self_test();save(ROOT/'closed_loop_classifier_preflight.json',r);print(json.dumps(r,indent=2));return
 result=collect([EVAL] if args.only_d0 else None);out=ROOT/args.label;assert not out.exists();out.mkdir();save(out/'failure_evidence.json',result);text=['闭环边界证据分型：'+result['status'],'阶段 | 完成/计划 | 持15 | 严格成功 | 分型（未完成阶段不可当最终结论）']
 for stage,g in result['groups'].items():text.append(f"{stage} {g['status']} | {g['completed']}/{g['expected']} | {g['held15']} | {g['strict_success']} | {g['outcomes']}")
 text+=['','逐episode证据：']
 for r in result['rows']:
  best=r['front300']['best_closing_dual_contact_margin'] or r['front300']['best_planned_palm_margin'];brief='' if best is None else f"; pregrasp step{best['control_step']} {best['boundary']} palm_margin={best['fruit']['min_margin_m']*1000:.2f}mm axes={best['fruit']['violated_axes']}";text.append(f"{r['stage']} seed{r['scene_seed']}: {r['outcome']}; grasp={r['first_grasp_step']}, held15={r['first_held15_step']}{brief}")
 text+=['']+result['limitations']
 with(out/'report.txt').open('x')as f:f.write('\n'.join(text)+'\n')
 print(json.dumps(dict(status=result['status'],output=str(out),groups=result['groups']),indent=2))
if __name__=='__main__':main()
