"""Regenerate B8000 prefix, execute live teacher, then independent student oracle."""
import bootstrap
import argparse,json,time,traceback
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
from recovery_common import *
from recovery_teacher import LiveRecoveryTeacher
from treesim.orchard_action import OrchardActionAdapter,encode_window,decode_targets

DIMS=dict(ee_pos=3,ee_rotm=9,arm_joint=7,arm_joint_vel=7,gripper_pos=1)

def regenerate(candidate):
 folder=Path(candidate['source_folder']);meta=json.loads((folder/'complete.json').read_text());assert meta['student_sha256']==PROTOCOL['baseline']['sha256']
 scene=meta['scene'];assert scene in PROTOCOL['source_episodes'];assert sha(scene['annotation'])==scene['annotation_sha256']
 chunks=json.loads((folder/'chunks.json').read_text());steps=json.loads((folder/'steps.json').read_text());env=environment();obs,info=env.reset(seed=scene['seed']);strict=StrictPlacement();adapter=OrchardActionAdapter();loaded=-1;prefix_command_difference_max=0.;first_command_difference_step=None
 try:
  for row in steps[:candidate['step']]:
   if row['chunk']!=loaded:
    c=chunks[row['chunk']];adapter.targets=(np.asarray(c['position']),np.asarray(c['rotation']),np.asarray(c['width']));loaded=row['chunk']
   command=adapter.to_native(row['k'],obs)
   command_difference=float(np.max(np.abs(np.asarray(command['action'])-row['command']['action'])))
   prefix_command_difference_max=max(prefix_command_difference_max,command_difference)
   if command_difference>1e-4 and first_command_difference_step is None:first_command_difference_step=row['step']
   obs,_,_,truncated,info=env.step(**command);update_strict(env,strict,row['step'])
  t=truth(env);expected=candidate['truth'];checks={}
  for field in ['joint_pos','tcp_position','tcp_quaternion','width','fruit_pose','base_pose','palm_local_fruit']:
   difference=float(np.max(np.abs(np.asarray(t[field])-expected[field])));checks[field]=difference;assert difference<=1e-4,(field,difference)
  for field in ['held','detached','branch_breaks']:
   assert np.array_equal(t[field],expected[field]),field
  assert abs(t['sim_time']-expected['sim_time'])<1e-9
  assert np.max(np.abs(np.asarray(t['fruit_velocity'])-expected['fruit_velocity']))<=1e-3
  return env,obs,strict,dict(status='PASS',physical_max_absolute_differences=checks,held_detached_exact=True,simulation_time_exact=True,prefix_steps=candidate['step'],original_chunks_sha256=sha(folder/'chunks.json'),original_steps_sha256=sha(folder/'steps.json'),prefix_command_difference_max=prefix_command_difference_max,first_command_difference_step=first_command_difference_step)
 except BaseException:env.close();raise

def measured(env):
 obs=env.get_obs()
 return dict(ee_pos=obs['tcp_pos_world'],ee_rotm=Rotation.from_quat(obs['tcp_quat_world']).as_matrix().ravel(),arm_joint=obs['joint_pos'][:7],arm_joint_vel=obs['joint_vel'][:7],gripper_pos=[obs['gripper_width']])
def held_physics(env):
 h=np.flatnonzero(env.sim.apples._held_host[:env.sim.apples.n]);assert len(h)<=1
 return None if not len(h) else int(h[0])
def strict_teacher(env,tracker,step):
 fs=fruit_snapshot(env);held=held_physics(env)
 return tracker.update(step,held,fs['detached_apple_ids'],fs['in_bucket_apple_ids'],bool(fs['in_bucket_apple_ids']))
def healthy(env,base,fruit):
 t=truth(env)
 if t['branch_breaks']:return 'branch_break'
 if int(np.count_nonzero(t['detached']))!=int(t['detached'][fruit]):return 'incidental_detach'
 if np.linalg.norm(t['base_pose'][:3]-base[:3])>=1e-3:return 'base_drift'
 if abs((Rotation.from_quat(t['base_pose'][3:])*Rotation.from_quat(base[3:]).inv()).magnitude())>=1e-3:return 'base_rotation'
 held=held_physics(env)
 if held is not None and held!=fruit:return 'wrong_apple_identity'
 arrays=[env.sim.body_q_np(),env.sim.state_0.body_qd.numpy(),env.sim.joint_q_np()]
 if not all(np.isfinite(a).all() for a in arrays):return 'nonfinite'
 return None

def teacher(candidate,directory):
 env,obs,strict,prefix_check=regenerate(candidate);fruit=int(env._reset_stance['apple_index']);base=env.sim.body_q_np()[env.chassis].copy();states={k:[] for k in DIMS};images=[[],[]];times=[];phases=[];steps=[];reason='teacher_budget';initial=truth(env);started=time.monotonic()
 try:
  # Save actual physics at takeover for independent replay comparison; no restore.
  with (directory/'takeover_physics.npz').open('wb') as f:np.savez_compressed(f,body_q=env.sim.state_0.body_q.numpy(),body_qd=env.sim.state_0.body_qd.numpy(),joint_q=env.sim.state_0.joint_q.numpy(),joint_qd=env.sim.state_0.joint_qd.numpy())
  source=json.loads(Path(json.loads((Path(candidate['source_folder'])/'complete.json').read_text())['scene']['annotation']).read_text());profile=dict(source['orchardbench']['arm_motion_profile']);profile['active_phases']=None if profile['active_phases'] is None else tuple(profile['active_phases'])
  picker=LiveRecoveryTeacher(env,candidate['category'],profile)
  def record():
   env._refresh_obs();obs=env.get_obs();v=measured(env)
   for k,a in v.items():
    a=np.asarray(a,dtype=float);assert a.shape==(DIMS[k],) and np.isfinite(a).all();states[k].append(a.tolist())
   times.append(float(env.sim.sim_time));phases.append(picker.state)
   for j,key in enumerate(['rgb_static','rgb_wrist']):
    a=obs[key];assert a.shape==(144,192,3) and a.dtype==np.uint8 and a.var()>1;images[j].append(a.copy())
  record()
  # Teacher assumes control of commands only. Remove stale student bias FF;
  # original AutoPicker gravity-sag servo is retained. Physical arrays untouched.
  force=env.sim.control.joint_f.numpy();force[env.dof_indices[:7]]=0.;env.sim.control.joint_f.assign(force)
  env.driver._tq_host[picker.arm_tq]=picker._q_cmd;env.sim.control.joint_target_q.assign(env.driver._tq_host)
  for frame in range(1,1801):
   env.driver.update();env.sim.step();picker.update()
   if frame%2:continue
   record();step=candidate['step']+frame//2
   outcome=strict_teacher(env,strict,step);bad=healthy(env,base,fruit)
   steps.append(dict(frame=frame,student_prefix_step=candidate['step'],control_step=step,phase=picker.state,held=held_physics(env),strict=outcome,truth=truth(env),failure=bad))
   if bad:reason=bad;break
   if outcome['strict_success'] and any(x['apple_id']==fruit for x in outcome['strict_success_events']):reason='strict_success';break
   if picker.state=='FAILED':reason='teacher_'+str(picker.fail_reason);break
  success=reason=='strict_success' and len(times)>=31
  trajectory=dict(schema='orchard_xr0_recovery_R_v1',trajectory_type='recovery_success' if success else 'recovery_rejected',episode_id=candidate['candidate_id'],split='train',seed=candidate['scene_seed'],num_frames=len(times),record_fps=30,proprios=states,actions={k:states[k][1:]+states[k][-1:] for k in DIMS if k!='arm_joint_vel'},observations={},recovery=dict(debug_usage='NOT POLICY INPUT',recovery_category=candidate['category'],source_episode_id=candidate['source_episode_id'],source_scene_seed=candidate['scene_seed'],student_checkpoint_sha256=PROTOCOL['baseline']['sha256'],student_inference_seed=candidate['inference_seed'],student_control_step=candidate['step'],student_chunk_index=candidate['chunk'],student_target_index=candidate['k'],takeover=initial,teacher_initial_state=picker.initial_state,initialization_audit=picker.initialization_audit,teacher_final_result=reason,prefix_regeneration=prefix_check),orchardbench=dict(debug_usage='NOT POLICY INPUT',timestamps=times,phase_by_frame=phases))
  result=dict(status='PASS' if success else 'REJECT',reason=reason,frames=len(times),strict=strict.summary(),initialization_audit=picker.initialization_audit,wall_seconds=time.monotonic()-started,prefix_regeneration=prefix_check)
  write(directory/'teacher_result.json',result);write(directory/'teacher_steps.json',steps)
  if success:
   encoder=module(ORCHARD/'scripts/collect_autopicker_dataset.py','r_encoder')
   for j,key in enumerate(['ego','wrist_left']):
    path=directory/f'{key}.mp4';encoder.encode(path,images[j]);trajectory['observations'][key]=[dict(path=str(path))]
  write(directory/'trajectory.json',trajectory)
  return trajectory,result
 finally:env.close()

def replay(candidate,traj,directory):
 env,obs,strict,regeneration=regenerate(candidate);base=env.sim.body_q_np()[env.chassis].copy();fruit=int(env._reset_stance['apple_index']);takeover=np.load(directory/'takeover_physics.npz');differences={};steps=[];reason='episode_budget';targets=0;timeouts=0;ikfailed=0
 try:
  for name in ['body_q','body_qd','joint_q','joint_qd']:
   current=getattr(env.sim.state_0,name).numpy();differences[name]=float(np.max(np.abs(current-takeover[name])))
   assert differences[name]<=(1e-4 if name.endswith('_q') else 1e-3),(name,differences[name])
  n=traj['num_frames'];p=traj['proprios'];chunks=[];indices=[]
  for first in range(0,n,30):
   anchor=min(first,n-30);adapter=OrchardActionAdapter();anchor_obs=dict(tcp_pos_world=np.asarray(p['ee_pos'][anchor]),tcp_quat_world=Rotation.from_matrix(np.asarray(p['ee_rotm'][anchor]).reshape(3,3)).as_quat());a=encode_window(traj,anchor);adapter.set_denormalized_chunk(a,anchor_obs)
   xyz,rot,width=adapter.targets;end=min(first+30,n);sl=slice(first-anchor,end-anchor)
   assert np.allclose(xyz[sl],traj['actions']['ee_pos'][first:end],atol=2e-6,rtol=0)
   assert np.allclose(rot[sl],np.asarray(traj['actions']['ee_rotm'][first:end]).reshape(-1,3,3),atol=2e-6,rtol=0)
   assert np.allclose(width[sl],np.asarray(traj['actions']['gripper_pos'][first:end]).ravel(),atol=2e-6,rtol=0)
   chunks.append((first-anchor,end-anchor,adapter));indices.extend(range(first,end))
  assert indices==list(range(n));chunkidx=0;k=chunks[0][0];dwell=0
  for step in range(candidate['step']+1,901):
   first,stop,adapter=chunks[chunkidx];xyz,rot,width=adapter.targets;terminal=chunkidx==len(chunks)-1 and k>=stop
   if terminal:k=stop-1
   cmd=adapter.to_native(k,obs);obs,_,_,truncated,info=env.step(**cmd);dwell+=1;outcome=update_strict(env,strict,step);adv,pe,re=advance(obs,xyz,rot,k,dwell);bad=healthy(env,base,fruit);ikfailed+=int(info['ik_failed'])
   steps.append(dict(step=step,chunk=chunkidx,k=k,terminal_hold=terminal,position_error=pe,rotation_error=re,dwell=dwell,advance=adv,held=env._held,truth=truth(env),strict=outcome,ik_failed=info['ik_failed'],failure=bad))
   if bad:reason=bad;break
   if outcome['strict_success'] and any(x['apple_id']==fruit for x in outcome['strict_success_events']):reason='strict_success';break
   if not terminal and adv:
    targets+=1;timeouts+=int(dwell>=30 and (pe>.01 or re>.08));k+=1;dwell=0
    if k==stop and chunkidx<len(chunks)-1:chunkidx+=1;k=chunks[chunkidx][0]
   if truncated:break
  result=dict(status='PASS' if reason=='strict_success' else 'REJECT',reason=reason,strict=strict.summary(),takeover_regeneration=regeneration,takeover_physics_max_absolute_differences=differences,control_steps=len(steps),total_episode_steps=env.control_steps,teacher_targets_consumed=targets,dwell_timeouts=timeouts,ik_failed_steps=ikfailed,action_roundtrip='PASS',next_state_indexing='PASS',same_student_controller=True,position_tolerance_m=.01,rotation_tolerance_rad=.08,max_dwell=30)
  write(directory/'oracle_result.json',result);write(directory/'oracle_steps.json',steps);return result
 finally:env.close()

def collect(candidate):
 directory=ROOT/'recoveries'/candidate['candidate_id'];directory.mkdir(parents=True,exist_ok=True)
 if (directory/'decision.json').exists():return json.loads((directory/'decision.json').read_text())
 write(directory/'candidate.json',candidate);print('CANDIDATE',candidate['candidate_id'],flush=True)
 try:
  if (directory/'teacher_result.json').exists():
   t=json.loads((directory/'trajectory.json').read_text());teacher_result=json.loads((directory/'teacher_result.json').read_text())
  else:t,teacher_result=teacher(candidate,directory)
  oracle=replay(candidate,t,directory) if teacher_result['status']=='PASS' else None
  accepted=teacher_result['status']=='PASS' and oracle is not None and oracle['status']=='PASS'
  result=dict(candidate_id=candidate['candidate_id'],category=candidate['category'],source_episode_id=candidate['source_episode_id'],inference_seed=candidate['inference_seed'],student_control_step=candidate['step'],accepted=accepted,teacher_status=teacher_result['status'],oracle_status=None if oracle is None else oracle['status'],reason=teacher_result['reason'] if teacher_result['status']!='PASS' else oracle['reason'],annotation=str(directory/'trajectory.json'),num_frames=t['num_frames'],legal_anchors=list(range(min(29,t['num_frames']-30)+1)) if accepted else [])
  if accepted:
   t['recovery']['oracle_replay_result']=oracle;write(directory/'trajectory.json',t);result['annotation_sha256']=sha(directory/'trajectory.json')
 except Exception as e:
  result=dict(candidate_id=candidate['candidate_id'],category=candidate['category'],source_episode_id=candidate['source_episode_id'],accepted=False,reason='engineering_exception',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
 write(directory/'decision.json',result);append(ROOT/'candidate_decisions.jsonl',result);print('DECISION',candidate['candidate_id'],result['accepted'],result['reason'],flush=True);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--per-category',type=int,default=2);p.add_argument('--category',choices=['R1','R2','R3']);a=p.parse_args();protection_check()
 candidates=[]
 for path in sorted((ROOT/'student_sources').glob('*/selected.json')):candidates+=json.loads(path.read_text())
 # Source cohort order is preserved; no validation scenes or new random seeds.
 order={e['episode_id']:i for i,e in enumerate(PROTOCOL['source_episodes'])};candidates.sort(key=lambda c:(c['inference_seed'],order[c['source_episode_id']],c['category']))
 counts=Counter()
 for path in (ROOT/'recoveries').glob('*/decision.json'):
  d=json.loads(path.read_text())
  if d['accepted']:counts[d['category']]+=1
 for c in candidates:
  if a.category and c['category']!=a.category:continue
  target=a.per_category if a.per_category else PROTOCOL['pilot_targets'][c['category']]
  if counts[c['category']]>=target:continue
  d=collect(c)
  if d['accepted'] and not (counts[c['category']]>=target):
   # Existing completed candidate already counted at initialization.
   counts=Counter(json.loads(p.read_text())['category'] for p in (ROOT/'recoveries').glob('*/decision.json') if json.loads(p.read_text())['accepted'])
  write(ROOT/'collection_status.json',dict(pid=__import__('os').getpid(),accepted_counts=dict(counts),current_candidate=c['candidate_id'],time=time.time()))
 print('ACCEPTED',dict(counts),flush=True);protection_check()
if __name__=='__main__':main()
