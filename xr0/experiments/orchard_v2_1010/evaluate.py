from common import *
import importlib.util
if importlib.util.find_spec('newton') is None:
 sys.path.append(str(ORCHARD/f'.pixi/envs/default/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'))
import argparse,gzip,time,traceback
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
from collections import Counter
from collect_student_native import truth,strict_mod,serial
from treesim.vla_env import OrchardVLAEnv,VLAEnvConfig
from treesim.student_native_expert import StudentNativeExpert
from treesim.orchard_action import native_command,decode_targets,state_vector,action_mask,prepare_rgb
sys.path.insert(0,str(ORCHARD/'experiments/initialization_prior_1009'))
from scenes import reset_override

class Student:
 def __init__(self,step):
  import torch
  from train import construct,overlay,seed
  from dataset import collator
  self.torch=torch;torch.set_num_threads(1);seed(42);self.model,_=construct()
  p=ROOT/'checkpoints'/f'step_{step:06d}.pt';ckpt=torch.load(p,map_location='cpu',weights_only=False);assert ckpt['step']==step;overlay(self.model,ckpt['trainable']);del ckpt
  self.model.to('cuda').eval();self.collate=collator();s=json.loads((ROOT/'action_stats.json').read_text());self.mean=np.array(s['mean']);self.std=np.array(s['std']);self.sha=sha(p)
 def plan(self,obs):
  from mibot.data.datasets.orchardbench_dataset import policy_messages
  torch=self.torch
  sample=dict(messages=policy_messages([prepare_rgb(Image.fromarray(obs[k])) for k in ['rgb_static','rgb_wrist']]),state=torch.from_numpy(state_vector(obs['tcp_pos_world'],R.from_quat(obs['tcp_quat_world']).as_matrix(),obs['gripper_width'],obs['joint_pos'][:7])),action=torch.zeros((30,32)),action_mask=torch.from_numpy(action_mask()))
  batch={k:v.to('cuda') for k,v in self.collate([sample]).items()}
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):a=self.model.generate(batch)[0].float().cpu().numpy()
  a=a*(self.std+1e-6)+self.mean;p,r,w=decode_targets(a,obs['tcp_pos_world'],R.from_quat(obs['tcp_quat_world']).as_matrix())
  return [dict(position=p[k],rotation=r[k],width=float(w[k])) for k in range(5)],a

def rollout(scene,init,student,label):
 folder=ROOT/'closed_loop'/label/f"{scene['seed']}_{init}";folder.mkdir(parents=True,exist_ok=True)
 if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
 previous=list(folder.glob('*.jsonl.gz'))
 if previous:
  archive=folder/f'interrupted_{time.time_ns()}';archive.mkdir()
  for old in previous:old.rename(archive/old.name)
 env=strict_mod.make_continuing_env(OrchardVLAEnv)(VLAEnvConfig(max_control_steps=2100));strict=strict_mod.StrictPlacement();expert=StudentNativeExpert() if student is None else None
 rows=[];releases=[];min_distance=None;phase='REACH';reason='episode_budget';start=time.monotonic();entered={'REACH':0};legacy=False
 try:
  obs,info=reset_override(env,scene,{'G0':0.,'G-':-.08,'G+':.08}[init]);initial=truth(env);target=initial['target_id']
  if student and not getattr(student,'remote',False):
   from train import seed
   seed(42)
  with gzip.open(folder/'steps.jsonl.gz','wt') as stream,gzip.open(folder/'chunks.jsonl.gz','wt') as chunks:
   while env.control_steps<2100:
    if expert:
     t=truth(env);block=expert.plan(obs,t,strict.summary()['fruit_chains'].get(str(target),{}))
     if not block:reason=expert.failure or 'expert_done';break
    else:
     block,action=student.plan(obs);chunks.write(json.dumps(dict(step=env.control_steps,action=action,anchor_position=obs['tcp_pos_world'],anchor_quat=obs['tcp_quat_world']),default=serial)+'\n')
    for command in block:
     before=env.control_steps;native=native_command(command['position'],command['rotation'],command['width'],obs['tcp_pos_world'],R.from_quat(obs['tcp_quat_world']).as_matrix())
     obs,_,_,truncated,info=env.step(**native);assert env.control_steps==before+1
     post=truth(env);nr=len(strict.release_events);strict.update(before+1,post['held'],post['detached'],post['in_bucket'],info['success']);legacy|=bool(info['success'])
     chains=strict.summary()['fruit_chains'];held=post['held'];chain=chains.get(str(held),{}) if held is not None else {}
     # Student has no FSM. These labels are passive event/geometry proxies only.
     distance=float(np.linalg.norm(np.array(post['fruit_position'])-obs['tcp_pos_world']))
     if expert:phase=command['phase']
     elif strict.release_events:phase='DROP'
     elif chain.get('first_detach_after_grasp_step') is not None:phase='TRANSPORT'
     elif chain.get('held15_step') is not None:phase='PULL'
     elif held is not None or distance<.045:phase='GRASP'
     else:phase='REACH'
     entered.setdefault(phase,before+1)
     if held is not None:
      pos=env.sim.body_q_np()[env.tm.apple_bodies[held],:3];d=float(np.linalg.norm(pos-post['bucket_position']));min_distance=d if min_distance is None else min(min_distance,d)
     for event in strict.release_events[nr:]:
      from treesim import robot
      pos=env.sim.body_q_np()[env.tm.apple_bodies[event['apple_id']],:3];delta=R.from_quat(post['base_pose'][3:]).inv().apply(pos-post['bucket_position']);dist=float(np.linalg.norm(delta));reasonable=dist<=.15 and abs(delta[0])<robot._BUCKET_HALF and abs(delta[1])<robot._BUCKET_HALF
      releases.append(dict(**event,fruit_bucket_distance=dist,offset_base=delta.tolist(),reasonable_release=bool(reasonable)))
     row=dict(step=before+1,phase_diagnostic=phase,held=held,detached=post['detached'],in_bucket=post['in_bucket'],legacy_success=bool(info['success']),ik_failed=bool(info['ik_failed']),clipped=bool(info['action_clipped']),tracking_position=float(np.linalg.norm(command['position']-obs['tcp_pos_world'])),tracking_rotation=float((R.from_matrix(command['rotation'])*R.from_quat(obs['tcp_quat_world']).inv()).magnitude()),command=command,native=native,tcp=obs['tcp_pos_world'],planned_fruit_position=post['fruit_position'])
     stream.write(json.dumps(row,default=serial,allow_nan=False)+'\n');rows.append({k:row[k] for k in ['step','phase_diagnostic','ik_failed','clipped','tracking_position','tracking_rotation']})
     if not post['finite']:reason='nonfinite_physics';break
     if strict.success_events:reason='strict_success';break
     if truncated:break
    if env.control_steps%100==0:write(folder/'progress.json',dict(step=env.control_steps,elapsed=time.monotonic()-start))
    if reason!='episode_budget' or truncated:break
  summary=strict.summary();chains=list(summary['fruit_chains'].values())
  def first(key):return min([v[key] for v in chains if v[key] is not None],default=None)
  durations=dict(Counter(r['phase_diagnostic'] for r in rows));timeout={p:durations.get(p,0)>=int(sec*30) for p,sec in [('REACH',12),('GRASP',12),('PULL',8),('TRANSPORT',45),('DROP',8)]}
  result=dict(label=label,checkpoint_sha256=student.sha if student else None,scene_seed=scene['seed'],initialization=init,initialization_valid=scene['initializations'][init]['initialization_valid'],common_valid=all(x['initialization_valid'] for x in scene['initializations'].values()),initial_geometry=scene['initializations'][init],planned_target=target,strict=summary,strict_success=summary['strict_success'],strict_success_planned_fruit=any(e['apple_id']==target for e in strict.success_events),legacy_success=legacy,entered_grasp='GRASP' in entered,held15=first('held15_step') is not None,detach=first('first_detach_after_grasp_step') is not None,actual_release=bool(releases),reasonable_release=any(e['reasonable_release'] for e in releases),stable_bucket=any(v['max_stable_bucket_steps']>=60 for v in chains),first_grasp_step=first('first_grasp_step'),release_step=min([e['step'] for e in releases],default=None),success_step=first('strict_success_step'),phase_steps=durations,phase_first_step=entered,phase_timeout_diagnostic=timeout,phase_definition='expert FSM' if expert else 'passive post-state proxy: release > detached-held > held15 > held or planned-target TCP distance<45mm; never input or termination',termination_reason=reason,steps=env.control_steps,ik_failure_rate=float(np.mean([r['ik_failed'] for r in rows])) if rows else None,clipping_rate=float(np.mean([r['clipped'] for r in rows])) if rows else None,tracking_position_p95=float(np.percentile([r['tracking_position'] for r in rows],95)) if rows else None,tracking_rotation_p95=float(np.percentile([r['tracking_rotation'] for r in rows],95)) if rows else None,min_held_fruit_bucket_distance=min_distance,release_diagnostics=releases,wall_seconds=time.monotonic()-start)
  write(folder/'result.json',result);print('ROLLOUT',label,scene['seed'],init,result['strict_success'],flush=True);return result
 except BaseException as e:
  write(folder/f'error_{time.time_ns()}.json',dict(error=str(e),traceback=traceback.format_exc(),scene_seed=scene['seed'],initialization=init));raise
 finally:env.close()

class RemoteStudent:
 remote=True
 def __init__(self,worker,requests,responses,digest):self.worker=worker;self.requests=requests;self.responses=responses;self.sha=digest;self.key=None
 def plan(self,obs):
  self.requests.put(('predict',self.worker,self.key,obs));kind,payload=self.responses.get()
  if kind!='prediction':raise RuntimeError(payload)
  return payload

def worker_loop(worker,jobs,requests,responses,label,digest,expert):
 proxy=None if expert else RemoteStudent(worker,requests,responses,digest)
 try:
  while True:
   task=jobs.get()
   if task is None:break
   scene,init=task
   if proxy:proxy.key=f"{scene['seed']}_{init}"
   result=rollout(scene,init,proxy,label);requests.put(('result',worker,result))
  requests.put(('finished',worker))
 except BaseException:
  requests.put(('error',worker,traceback.format_exc()))

def main():
 p=argparse.ArgumentParser();p.add_argument('--step',type=int);p.add_argument('--expert',action='store_true');a=p.parse_args();assert a.expert!=(a.step is not None)
 model=None if a.expert else Student(a.step);label='expert' if a.expert else f'step_{a.step:06d}'
 import multiprocessing,queue
 context=multiprocessing.get_context('spawn');requests=context.Queue();jobs=context.Queue();responses=[context.Queue() for _ in range(4)]
 for scene in json.loads((ROOT/'scenes.json').read_text())['scenes']:
  for init in ['G0','G-','G+']:jobs.put((scene,init))
 for _ in range(4):jobs.put(None)
 workers=[context.Process(target=worker_loop,args=(i,jobs,requests,responses[i],label,model.sha if model else None,a.expert)) for i in range(4)]
 for w in workers:w.start()
 results=[];finished=0;states={}
 try:
  while finished<4:
   try:message=requests.get(timeout=10)
   except queue.Empty:
    if any(w.exitcode not in (None,0) for w in workers):raise RuntimeError('simulation worker exited unexpectedly')
    continue
   kind,worker,*payload=message
   if kind=='predict':
    from train import seed,rng_state,restore_rng
    key,obs=payload
    if key in states:restore_rng(states[key])
    else:seed(42)
    prediction=model.plan(obs);states[key]=rng_state();responses[worker].put(('prediction',prediction))
   elif kind=='result':results.append(payload[0])
   elif kind=='finished':finished+=1
   elif kind=='error':raise RuntimeError(payload[0])
 finally:
  for w in workers:
   if w.is_alive() and finished<4:w.terminate()
  for w in workers:w.join()
 assert len(results)==48
 write(ROOT/'closed_loop'/label/'summary.json',dict(results=results,execution='four simulation workers; one shared VLM, serial batch1 inference; independent per-scene RNG restored on every request',aggregate={g:{k:sum(bool(r[k]) for r in results if r['initialization']==g) for k in ['strict_success','legacy_success','held15','detach','actual_release','reasonable_release','stable_bucket']} for g in ['G0','G-','G+']}))
if __name__=='__main__':main()
