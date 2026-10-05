"""Optional next R iteration: real R4000 pre-release states after saved B prefix.
No state restoration. Composite native-controller rollout regenerates takeover.
Truth is used by teacher/filter/debug only. Original datasets remain read-only.
"""
import bootstrap
import argparse,json,time,traceback,fcntl
from pathlib import Path
import numpy as np
from treesim.orchard_action import OrchardActionAdapter,native_command
from scipy.spatial.transform import Rotation
from recovery_common import *
import collect_recovery as collector
p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args()
OUT=ROOT/'evaluation_followup/onpolicy_R3';CURRENT=ROOT/'training/checkpoints/step_4000_trainable.pt';DIAG=ROOT/'evaluation_followup/version_01/takeovers/R4000'

def prepare_sources():
 OUT.mkdir(parents=True,exist_ok=True);candidates=[]
 for result_path in sorted(DIAG.glob('*/result.json')):
  result=json.loads(result_path.read_text())
  if result['category']!='R2' or result['strict_success'] is None:continue
  oldcandidate=json.loads((ROOT/'recoveries'/result['candidate']/'candidate.json').read_text());Rsteps=json.loads((result_path.parent/'steps.json').read_text());Rchunks=json.loads((result_path.parent/'chunks.json').read_text());oldfolder=Path(oldcandidate['source_folder']);oldmeta=json.loads((oldfolder/'complete.json').read_text());scene=oldmeta['scene']
  assert scene['episode_id'] in {s['episode_id'] for s in PROTOCOL['source_episodes']} and scene['episode_id'] not in PROTOCOL['protected_episode_ids']
  assert result['checkpoint_sha256']==sha(CURRENT)
  eligible=[r for r in Rsteps if r['step']<=720 and r['truth']['held']==[r['truth']['fruit_id']] and r['truth']['detached'][r['truth']['fruit_id']] and r['truth']['fruit_bucket_distance']<=.25 and not r['truth']['branch_breaks']]
  assert eligible,('No bounded near-bucket candidate',result['candidate'])
  start=eligible[0];prefix_steps=json.loads((oldfolder/'steps.json').read_text())[:oldcandidate['step']];oldchunks=json.loads((oldfolder/'chunks.json').read_text());count=max(r['chunk'] for r in prefix_steps)+1;chunks=oldchunks[:count];previous=result['takeover']
  for rc in Rchunks:chunks.append(dict(chunk=len(chunks),position=rc['position'],rotation=rc['rotation'],width=rc['width'],at_step=rc['step']-1,source_policy_sha256=sha(CURRENT)))
  steps=list(prefix_steps)
  for row in Rsteps:
   if row['step']>start['step']:break
   chunk=chunks[count+row['chunk']];k=row['k'];command=native_command(chunk['position'][k],chunk['rotation'][k],chunk['width'][k],previous['tcp_position'],Rotation.from_quat(previous['tcp_quaternion']).as_matrix())
   steps.append(dict(step=row['step'],chunk=count+row['chunk'],k=k,dwell=row['dwell'],advance=row['advance'],command=command,truth=row['truth']));previous=row['truth']
  assert [r['step'] for r in steps]==list(range(1,start['step']+1))
  cid=f"{scene['episode_id']}_Bprefix_R4000_rng42_R3_s{start['step']:04d}";folder=OUT/'student_sources'/cid
  lineage=[dict(policy='B8000',sha256=PROTOCOL['baseline']['sha256'],start_step=0,end_step=oldcandidate['step']),dict(policy='R4000',sha256=sha(CURRENT),start_step=oldcandidate['step']+1,end_step=start['step'])]
  candidate=dict(category='R3',step=start['step'],chunk=count+start['chunk'],k=start['k'],rule='R4000_actual_stable_detached_held_first_near_bucket_le_025_before720',truth=start['truth'],source_episode_id=scene['episode_id'],scene_seed=scene['seed'],inference_seed=42,source_folder=str(folder),candidate_id=cid,student_policy_sha256=sha(CURRENT),prefix_policy_lineage=lineage)
  if not folder.exists():
   write(folder/'chunks.json',chunks);write(folder/'steps.json',steps);write(folder/'complete.json',dict(scene=scene,student_sha256=sha(CURRENT),prefix_policy_lineage=lineage,source_R_trace_sha256=sha(result_path.parent/'steps.json'),source_R_chunks_sha256=sha(result_path.parent/'chunks.json'),original_B_folder=str(oldfolder),candidate=candidate));write(folder/'candidate.json',candidate)
  else:assert json.loads((folder/'candidate.json').read_text())==candidate
  candidates.append(candidate)
 write(OUT/'source_selection.json',dict(candidates=candidates,protected_scenes_excluded=True,policy_input_unchanged=True,source_policy='Actual R4000 continuation after B8000 prefix; training-source diagnostic scenes only',not_clean_R_reset_rollout=True,no_teleport_or_snapshot_restore=True));return candidates

def regenerate(candidate):
 folder=Path(candidate['source_folder']);meta=json.loads((folder/'complete.json').read_text());assert meta['student_sha256']==sha(CURRENT);scene=meta['scene'];assert sha(scene['annotation'])==scene['annotation_sha256'];assert scene in PROTOCOL['source_episodes'];chunks=json.loads((folder/'chunks.json').read_text());steps=json.loads((folder/'steps.json').read_text());env=environment();obs,info=env.reset(seed=scene['seed']);tracker=StrictPlacement();adapter=OrchardActionAdapter();loaded=-1;maximum=0.
 try:
  for row in steps[:candidate['step']]:
   if row['chunk']!=loaded:
    c=chunks[row['chunk']];adapter.targets=(np.asarray(c['position']),np.asarray(c['rotation']),np.asarray(c['width']));loaded=row['chunk']
   command=adapter.to_native(row['k'],obs);maximum=max(maximum,float(np.max(np.abs(np.asarray(command['action'])-row['command']['action']))));obs,_,_,trunc,info=env.step(**command);update_strict(env,tracker,row['step'])
  t=truth(env);expected=candidate['truth'];checks={}
  for field in ['joint_pos','tcp_position','tcp_quaternion','width','fruit_pose','base_pose','palm_local_fruit']:
   difference=float(np.max(np.abs(np.asarray(t[field])-expected[field])));checks[field]=difference;assert difference<=1e-4,(field,difference)
  assert np.max(np.abs(np.asarray(t['fruit_velocity'])-expected['fruit_velocity']))<=1e-3
  for field in ['held','detached','branch_breaks']:assert np.array_equal(t[field],expected[field]),field
  assert abs(t['sim_time']-expected['sim_time'])<1e-9
  return env,obs,tracker,dict(status='PASS',physical_max_absolute_differences=checks,held_detached_exact=True,simulation_time_exact=True,prefix_steps=candidate['step'],composite_B_then_R_prefix=True,prefix_policy_lineage=meta['prefix_policy_lineage'],prefix_command_difference_max=maximum,original_chunks_sha256=sha(folder/'chunks.json'),original_steps_sha256=sha(folder/'steps.json'))
 except BaseException:env.close();raise

def run():
 protection_check();lock=(ROOT/'.gpu_training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);candidates=prepare_sources();collector.regenerate=regenerate;decisions=[]
 for candidate in candidates:
  directory=OUT/'recoveries'/candidate['candidate_id'];directory.mkdir(parents=True,exist_ok=True);write(directory/'candidate.json',candidate)
  if (directory/'decision.json').exists():decisions.append(json.loads((directory/'decision.json').read_text()));continue
  try:
   trajectory,result=collector.teacher(candidate,directory)
   trajectory['recovery'].update(student_checkpoint_sha256=sha(CURRENT),student_policy='R4000',prefix_policy_lineage=candidate['prefix_policy_lineage'],source_stage='Actual learner continuation after B prefix',sampling_role='new actual near-bucket student pre-release state; no canonical suffix labels')
   write(directory/'trajectory.json',trajectory)
   oracle=collector.replay(candidate,trajectory,directory) if result['status']=='PASS' else None
   accepted=result['status']=='PASS' and oracle is not None and oracle['status']=='PASS'
   if accepted:trajectory['recovery']['oracle_replay_result']=oracle;write(directory/'trajectory.json',trajectory)
   decision=dict(candidate_id=candidate['candidate_id'],category='R3',source_episode_id=candidate['source_episode_id'],inference_seed=42,student_control_step=candidate['step'],accepted=accepted,teacher_status=result['status'],oracle_status=None if oracle is None else oracle['status'],reason=result['reason'] if result['status']!='PASS' else oracle['reason'],annotation=str(directory/'trajectory.json'),num_frames=trajectory['num_frames'],legal_anchors=list(range(min(29,trajectory['num_frames']-30)+1)) if accepted else [],student_policy_sha256=sha(CURRENT),sampling_subtype='R4000_near_bucket_pre_release')
   if accepted:decision['annotation_sha256']=sha(directory/'trajectory.json')
  except Exception as e:decision=dict(candidate_id=candidate['candidate_id'],category='R3',source_episode_id=candidate['source_episode_id'],accepted=False,reason='engineering_exception',error=str(e),traceback=traceback.format_exc())
  write(directory/'decision.json',decision);append(OUT/'candidate_decisions.jsonl',decision);decisions.append(decision);print(json.dumps(decision),flush=True)
 write(OUT/'status.json',dict(status='COMPLETE',accepted=sum(d['accepted'] for d in decisions),candidate=len(decisions),pid=__import__('os').getpid()));protection_check()
if __name__=='__main__':
 if a.run:run()
 else:print('No collection launched; --run required')
