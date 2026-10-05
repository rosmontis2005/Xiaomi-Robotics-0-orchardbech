"""Bounded B8000 actual rollout mining. All eligible boundaries and decisions saved."""
import bootstrap
import argparse,time
from collections import Counter
import numpy as np
from recovery_common import *
from treesim.orchard_action import OrchardActionAdapter
observers=module(XR0/'diagnosis_1003/observers.py','r_observers')

def mine(policy,scene,seed):
 folder=ROOT/'student_sources'/f"{scene['episode_id']}_rng{seed}";folder.mkdir(parents=True,exist_ok=True)
 if (folder/'complete.json').exists():return
 env=environment();candidates={k:[] for k in ['R1','R2','R3']};rows=[];chunks=[];strict=StrictPlacement();heldrun=0;previous=None;distances=[];chunk=-1;reason='budget'
 started=time.monotonic()
 try:
  obs,info=env.reset(seed=scene['seed']);initial=truth(env);write(folder/'initial.json',dict(scene=scene,inference_seed=seed,truth=initial,physics_digest=physics_digest(env)))
  assert info['seed']==scene['seed'];p=r=w=None;k=30;dwell=0
  for step in range(1,901):
   if k==30:
    normalized=policy.predict(obs,seed=seed);chunk+=1;k=0;dwell=0;p,r,w=policy.adapter.targets
    chunks.append(dict(chunk=chunk,at_step=step-1,normalized=normalized,position=p,rotation=r,width=w,anchor_position=obs['tcp_pos_world'],anchor_quaternion=obs['tcp_quat_world']))
   command=policy.adapter.to_native(k,obs)
   # Capture BEFORE a predicted unsafe open, without executing the release.
   pre=truth(env)
   if previous==pre['fruit_id'] and heldrun>=15 and pre['detached'][pre['fruit_id']] and pre['fruit_bucket_distance']>.15 and w[k]>=.075 and env._width_open_steps>=3:
    candidates['R3'].append(dict(category='R3',step=step-1,chunk=chunk,k=k,rule='imminent_unsafe_open',score=100.,truth=pre))
   obs,_,_,truncated,info=env.step(**command);dwell+=1
   t=truth(env);fruit=t['fruit_id'];held=env._held
   heldrun=heldrun+1 if held is not None and held==previous else 1 if held is not None else 0;previous=held
   distances.append(t['fruit_bucket_distance']);ss=update_strict(env,strict,step)
   snap=observers.snapshot_grasp(env);contact=next((v for v in snap['contact_apples'] if v['apple_id']==fruit),None)
   if held is None and not t['detached'][fruit] and t['tcp_fruit_distance']<=.09 and (env._gripper<0 or contact):
    score=(4 if contact and contact['both_fingers_contact'] else 2 if contact else 0)+(1 if env._gripper<0 else 0)-t['tcp_fruit_distance']
    candidates['R1'].append(dict(category='R1',step=step,chunk=chunk,k=k,rule='actual_near_contact_with_closing_or_contact',score=score,truth=t,contact=contact))
   if held==fruit and heldrun>=15 and t['fruit_bucket_distance']>.25:
    candidates['R2'].append(dict(category='R2',step=step,chunk=chunk,k=k,rule='stable_held_transport',score=1 if t['detached'][fruit] else 0,truth=t))
   if held==fruit and heldrun>=15 and t['detached'][fruit] and step>=350 and t['fruit_bucket_distance']>.15 and len(distances)>60 and distances[-61]-distances[-1]<.03:
    candidates['R3'].append(dict(category='R3',step=step,chunk=chunk,k=k,rule='late_transport_stall',score=10.,truth=t))
   adv,pe,re=advance(obs,p,r,k,dwell)
   rows.append(dict(step=step,chunk=chunk,k=k,dwell=dwell,advance=adv,position_error=pe,rotation_error=re,command=command,truth=t,held_run=heldrun,branch_breaks=info['branch_break_count'],strict=ss))
   if adv:k+=1;dwell=0
   if step%150==0:print(f"MINING {scene['episode_id']} rng{seed} step{step} held{held} candidates {dict((c,len(v)) for c,v in candidates.items())}",flush=True)
   if info['branch_break_count'] or int(np.count_nonzero(t['detached']))>int(t['detached'][fruit]):reason='physical_anomaly';break
   if ss['strict_success']:reason='strict_success';break
   if truncated:break
   if all(candidates[c] for c in ['R1','R2','R3']):reason='bounded_mining_all_categories_found';break
  selected=[];used=set()
  for category,values in candidates.items():
   valid=[v for v in values if v['step']>0 and not v['truth']['branch_breaks'] and int(np.count_nonzero(v['truth']['detached']))==int(v['truth']['detached'][v['truth']['fruit_id']])]
   if category=='R1':valid.sort(key=lambda c:(-c['score'],c['step']))
   else:valid.sort(key=lambda c:(-c['score'],c['step']))
   for v in valid:
    if v['step'] not in used:
     selected.append(dict(v,source_episode_id=scene['episode_id'],scene_seed=scene['seed'],inference_seed=seed,source_folder=str(folder),candidate_id=f"{scene['episode_id']}_rng{seed}_{category}_s{v['step']:04d}"));used.add(v['step']);break
  write(folder/'chunks.json',chunks);write(folder/'steps.json',rows);write(folder/'eligible_candidates.json',candidates);write(folder/'selected.json',selected)
  write(folder/'complete.json',dict(scene=scene,inference_seed=seed,student_sha256=PROTOCOL['baseline']['sha256'],steps=len(rows),termination=reason,eligible_counts={c:len(v) for c,v in candidates.items()},selected=[v['candidate_id'] for v in selected],wall_seconds=time.monotonic()-started,strict=strict.summary()))
  print('MINED',scene['episode_id'],[(v['category'],v['step']) for v in selected],flush=True)
 finally:env.close()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--start',type=int,default=0);parser.add_argument('--count',type=int,default=4);parser.add_argument('--seed',type=int,choices=[42,43],default=42);a=parser.parse_args()
 protection_check();policy=make_policy()
 for scene in PROTOCOL['source_episodes'][a.start:a.start+a.count]:mine(policy,scene,a.seed)
 protection_check()
if __name__=='__main__':main()
