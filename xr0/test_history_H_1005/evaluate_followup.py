"""Frozen R endpoint evaluation; reuse existing full30 controller and strict observer."""
import argparse,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--condition',choices=['R03-8k','H0','Hrepeat','Hhistory','Hanchor'],required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--offline',action='store_true');a=p.parse_args()
os.environ['H_RUN_DIR']=str(Path(__file__).resolve().parent/('r03_extension' if a.condition=='R03-8k' else a.condition))
import bootstrap
import argparse,json,time,os,sys,traceback,fcntl
from pathlib import Path
from collections import Counter
from scipy.spatial.transform import Rotation
import numpy as np
import torch
from recovery_common import ROOT,XR0,ORCHARD,BASE,STATS,PROCESSOR,sha,module,protection_check
from checkpoint_io import load_trainable_overlay
AB=XR0/'test_128_episode_AB_1003';D0=XR0/'test_grasp_1004/evaluation';bootstrap.AB=AB
sys.path.append(str(D0))
OUT=ROOT/'evaluation';OUT.mkdir(parents=True,exist_ok=True)
def write(path,data):
 path=Path(path);assert path.is_relative_to(OUT);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)
def lines(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def status(phase,**kw):write(OUT/'status.json',dict(status=phase,pid=os.getpid(),time=time.time(),**kw))
try:
 torch.set_num_threads(1);protection_check();lock=(ROOT/'.gpu_training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 d0=json.loads((D0/'protocol.json').read_text());selection=AB/'selection.json'
 assert sha(STATS)==d0['stats_sha256'] and sha(selection)==d0['selection_sha256']
 checkpoint=Path(a.checkpoint);payload=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True);step=payload['step'];assert step in (4000,8000)
 jobs=[]
 for stage in ['B30','regression_B30']:
  for prior in d0['jobs'][stage]:
   j=dict(prior);j.update(stage='R30' if stage=='B30' else 'regression_R30',provider=prior['provider'].replace('B8000',a.condition),job_id=prior['job_id'].replace('B8000',a.condition),baseline_job_id=prior['job_id']);jobs.append(j)
 protocol=dict(checkpoint=str(checkpoint),step=step,checkpoint_sha256=sha(checkpoint),selection_sha256=sha(selection),stats_sha256=sha(STATS),baseline_protocol_sha256=sha(D0/'protocol.json'),reach=d0['reach'],strict=d0['strict'],jobs=jobs,code_sha256=sha(Path(__file__)),no_fresh24=True,condition=a.condition,history_offsets=([0] if a.condition in ['R03-8k','H0'] else [0]*4 if a.condition=='Hrepeat' else [-15,-5,-1,0]),endpoint_selection='predeclared budget endpoint; no intermediate checkpoint selection',baseline='frozen matching D0 B8000 traces with exact reset checks')
 if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
 else:write(OUT/'protocol.json',protocol)
 logrows=lines(ROOT/'training/training_steps.jsonl');assert len(logrows)==(4000 if a.condition=='R03-8k' else step) and all(x['real_optimizer_update'] for x in logrows)
 audit=json.loads((ROOT/'data_audit.json').read_text())
 write(OUT/'training_audit.json',dict(status=json.loads((ROOT/'training/status.json').read_text()),checkpoint_sha256=sha(checkpoint),checkpoint_step=step,finite_parameters=all(bool(torch.isfinite(t).all()) for t in payload['trainable_state_dict'].values()),trainable_tensors=len(payload['trainable_state_dict']),sources=dict(Counter(x['source'] for x in logrows)),loss_by_source={s:dict(first500=float(np.mean([x['flow_loss'] for x in logrows[:500] if x['source']==s])),last500=float(np.mean([x['flow_loss'] for x in logrows[-500:] if x['source']==s]))) for s in sorted(Counter(x['source'] for x in logrows))},data_audit_sha256=sha(ROOT/'data_audit.json')))
 del payload
 status('MODEL_LOADING')
 from history_inputs import HistoryPolicy,CausalHistory
 policy=HistoryPolicy(str(BASE),str(PROCESSOR),str(STATS),mode='H0' if a.condition=='R03-8k' else a.condition);loaded=load_trainable_overlay(policy.model,checkpoint,base_checkpoint=BASE);assert loaded['step']==step
 policy.model.num_steps=5
 dtypes={group:dict(Counter(str(v.dtype) for n,v in policy.model.named_parameters() if n.startswith('vlm.')==flag)) for group,flag in [('vlm',True),('non_vlm',False)]};assert dtypes=={'vlm':{'torch.bfloat16':713},'non_vlm':{'torch.float32':219}}
 write(OUT/'load_manifest.json',dict(loaded=loaded,dtypes=dtypes))
 if a.offline:
  common=module(AB/'ab_common.py','R_offline_common');common.SELECTION=selection;common.ROOT=OUT
  cache=torch.load(AB/'eval_cache.pt',map_location='cpu',weights_only=True,mmap=True)
  if a.condition!='R03-8k':
   history=CausalHistory()
   for sample in cache['samples']:
    meta=sample['meta'];path=meta.get('annotation',str(ORCHARD/'data/orchard_v1_2650/filtered/json'/meta['split']/(meta['episode_id']+'.json')))
    sample['batch']['state']=torch.from_numpy(history.states(path,meta['frame'],a.condition))[None]
  common.evaluate(policy.model,cache['samples'],cache['mean'].numpy(),cache['std'].numpy(),step,'cuda:0',OUT/'offline',lambda n,total:status('OFFLINE',windows=n,total=total))
  del cache
  from late_offline import evaluate_late
  if a.condition=='R03-8k':
   historical=XR0/'test_recovery_R_1004/evaluation_followup/version_03/training/checkpoints/step_4000_trainable.pt'
   load_trainable_overlay(policy.model,historical,base_checkpoint=BASE);evaluate_late(policy,'H0',OUT/'late_offline_R03_4k.json');load_trainable_overlay(policy.model,checkpoint,base_checkpoint=BASE)
  evaluate_late(policy,'H0' if a.condition=='R03-8k' else a.condition,OUT/'late_offline.json')
 loop=module(D0/'reach_loop.py','R_frozen_reach');loop.ROOT=OUT;loop.load_runtime()
 # Experiment-only observation hook; super.step executes unchanged production.
 original_env=loop.OrchardVLAEnv
 class ObservedEnv(original_env):
  def reset(self,*args,**kw):
   obs,info=super().reset(*args,**kw);policy.reset_history(obs);return obs,info
  def step(self,*args,**kw):
   result=super().step(*args,**kw);policy.observe_control_step(result[0],self.control_steps);return result
 loop.OrchardVLAEnv=ObservedEnv

 helpers=module(AB/'closed_loop_1004/evaluate_ab.py','R_old_helpers');helpers.ROOT=OUT
 analyze=module(XR0/'diagnosis_1003/analyze_rollouts.py','R_analyzer');passive=module(AB/'closed_loop_1004/passive_metrics.py','R_passive')
 active={};base_line=loop.line;base_snapshot=loop.snapshot_grasp
 def line(stream,data):
  data.update(inference_seed=active['inference_seed'],job_id=active['job_id']);base_line(stream,data)
  if 'control_step' in data and data['control_step']%100==0:status('CLOSED_LOOP',job=active['job_id'],control_step=data['control_step'],completed=len(results))
 def snapshot(env):
  s=base_snapshot(env);idx=active['planned_apple_id'];body=int(env.tm.apple_bodies[idx]);s['planned_apple']=passive.planned_apple_observation(s,idx,body,env.sim.body_q_np()[body,:3])
  from treesim import robot
  cp,cr=env._chassis_pose();bucket=cp+cr.apply([robot._BUCKET_CENTER_X,0.,robot._CHASSIS_Z+robot._CHASSIS[2]+robot._BUCKET_WALL_H]);apple=env.sim.body_q_np()[body,:3];local=cr.inv().apply(apple-cp)
  s['placement_geometry']=dict(diagnostic_only=True,bucket_top_center_world=bucket.tolist(),planned_fruit_bucket_distance_m=float(np.linalg.norm(apple-bucket)),planned_fruit_base_local=local.tolist(),base_position=cp.tolist(),base_quaternion=cr.as_quat().tolist())
  ids=active.setdefault('observed_fruit_ids',set());ids.add(idx)
  if s['held_apple_id'] is not None:ids.add(int(s['held_apple_id']))
  poses=env.sim.body_q_np();s['fruit_bucket_geometry']={str(i):dict(world_position=poses[int(env.tm.apple_bodies[i]),:3].tolist(),distance_m=float(np.linalg.norm(poses[int(env.tm.apple_bodies[i]),:3]-bucket)),base_local=cr.inv().apply(poses[int(env.tm.apple_bodies[i]),:3]-cp).tolist()) for i in sorted(ids)}
  return s
 loop.line=line;loop.snapshot_grasp=snapshot
 results=[]
 for j in jobs:
  folder=OUT/'rollouts'/j['job_id'];active.clear();active.update(j);traj=json.loads(Path(j['scene']['annotation']).read_text());active['planned_apple_id']=int(traj['orchardbench']['fixed_base_expert']['selected_apple_debug_index'])
  if (folder/'completed.json').exists():results.append(json.loads((folder/'summary.json').read_text()));continue
  assert not folder.exists(),('Partial retained',folder)
  assert sha(j['scene']['annotation'])==j['scene']['annotation_sha256'];status('CLOSED_LOOP',job=j['job_id'],completed=len(results),control_step=0)
  bound=helpers.BoundSeedPolicy(policy,j['inference_seed'])
  r=loop.episode(bound,j['provider'],j['scene'],str(checkpoint),True,sha(selection),processed_targets=30,is_gt=False)
  assert r['termination']!='error'
  steps=lines(folder/'steps.jsonl');chunks=lines(folder/'chunks.jsonl');assert len(bound.calls)==r['replans']==len(chunks)
  metrics=analyze.analyze_episode(r,steps,chunks,folder);metrics.update(helpers.held_runs(steps))
  gtref=AB/'closed_loop_1004/rollouts'/f'gt_reach-conditioned_{j["scene"]["seed"]}'/'chunks.jsonl';metrics.update(passive.diagnostics(steps,chunks,lines(gtref)))
  metrics.update(strict_placement=r['strict_placement'],strict_success=r['strict_success'],legacy_success_ever=r['legacy_success_ever'])
  baseline=D0/'rollouts'/j['baseline_job_id'];assert json.loads((folder/'initial.json').read_text())['obs']==json.loads((baseline/'initial.json').read_text())['obs']
  assert r['scene_check_pass'] and metrics['observer_consistency']['mismatch_fields_total']==0 and metrics['summary_consistency']['mismatch_count']==0
  write(folder/'metrics.json',metrics);write(folder/'seed_delivery.json',dict(calls=bound.calls));write(folder/'history_delivery.json',dict(mode=policy.history.mode,updates=policy.history.updates,control_steps=r['control_steps'],calls=policy.history_calls));assert policy.history.updates==r['control_steps']+1;write(folder/'completed.json',dict(status='COMPLETE',checkpoint_sha256=sha(checkpoint),baseline_reset_exact=True,files_sha256={name:sha(folder/name) for name in ['summary.json','metrics.json','steps.jsonl','chunks.jsonl','initial.json']}));results.append(r)
  print(json.dumps(dict(event='JOB_COMPLETE',job=j['job_id'],strict=r['strict_success'],held15=metrics['same_fruit_held_at_least_15_steps'])),flush=True)
 if a.condition=='R03-8k':
  from inference_smoke import rollout_smoke
  write(OUT/'history_inference_smoke.json',rollout_smoke(policy,original_env,loop.CONFIG,jobs[0]['scene']))
 write(OUT/'results.json',results);status('COMPLETE',completed=len(results),strict=sum(r['strict_success'] for r in results))
except BaseException as e:status('FAILED',error=str(e),traceback=traceback.format_exc());raise
