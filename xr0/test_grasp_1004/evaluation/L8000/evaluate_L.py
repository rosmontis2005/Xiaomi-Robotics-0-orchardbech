#!/usr/bin/env python3
"""D0: fixed B8000, full30 versus processed5 reach, strict same-fruit placement."""
import bootstrap
import argparse
import ast
from collections import Counter
import fcntl
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT,AB,XR0,ORCHARD=bootstrap.ROOT,bootstrap.AB,bootstrap.XR0,bootstrap.ORCHARD
OLD=AB/'closed_loop_1004'
BASE=XR0/'outputs/orchard_v1_2650/project_orchardbench/orchard_v1_2650_frozen_vlm_seed42/epoch=0-step=10000.ckpt'
STATS=ORCHARD/'data/orchard_v1_2650/filtered/action_stats.json'
PROCESSOR=XR0.parent/'checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D'
SELECTION=AB/'selection.json'
PROTOCOL=ROOT/'protocol.json'
D0=ROOT.parent
TRAINING=bootstrap.EXPERIMENT/'training'
STAGES=['L30','L5','regression_L30']

def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

helpers=module(OLD/'evaluate_ab.py','frozen_ab_helpers')
sha=helpers.sha;save_new=helpers.save_new;atomic_json=helpers.atomic_json;log=helpers.log

def read_lines(path):
 with Path(path).open() as stream:return [json.loads(row) for row in stream if row.strip()]

def source_hashes():
 old=json.loads((OLD/'protocol.json').read_text())['source_sha256']
 for path,digest in old.items():assert sha(path)==digest,('Historical source changed',path)
 paths=set(old)|{str(p) for p in ROOT.glob('*.py')}
 return {p:sha(p) for p in sorted(paths)}

def prepare():
 core=module(D0/'reach_loop.py','L_cpu_frozen_core');assert core.ROOT==ROOT
 assert core.advance_decision(.01,.08,1)==(True,False,True,'reached')
 assert 'treesim.vla_env' not in sys.modules and 'torch' not in sys.modules
 d0=json.loads((D0/'protocol.json').read_text())
 jobs={}
 for stage,base in [('L30','B30'),('L5','B5'),('regression_L30','regression_B30')]:
  jobs[stage]=[]
  for prior in d0['jobs'][base]:
   item=dict(prior);item.update(stage=stage,provider=prior['provider'].replace('B8000','L8000'),job_id=prior['job_id'].replace('B8000','L8000'))
   jobs[stage].append(item)
 sources=dict(d0['source_sha256']);sources.update({str(p):sha(p) for p in ROOT.glob('*.py')})
 protocol=dict(schema='orchard_grasp_L8000_v2',created_unix=time.time(),selection_path=str(SELECTION),selection_sha256=sha(SELECTION),stats_path=str(STATS),stats_sha256=sha(STATS),source_sha256=sources,jobs=jobs,queue_order=STAGES,d0_protocol_sha256=sha(D0/'protocol.json'),checkpoint=dict(path=str(TRAINING/'arms/L/checkpoints/step_8000_trainable.pt'),step=8000,arm='L',sha256='verified_after_training_completion'),reach=d0['reach'],scheduler=d0['scheduler'],strict=d0['strict'],endpoint_selection='Predeclared L8000 only; never best checkpoint',model_episodes=18,primary_dev_episodes=16,auxiliary_regression_episodes=2,revision='Root predeclared same historical2010600 seed42/43 commit30 regression before its D0 outcomes and before L training; no outcome-based selection',baseline='Same current-run D0 B8000, each scene seed42 at matching processed30/5 scheduler',prior_control_sources_unchanged=True)
 save_new(PROTOCOL,protocol)
 for p,digest in sources.items():assert sha(p)==digest
 save_new(ROOT/'preflight.json',dict(status='PASS',cuda_initialized=False,protocol_sha256=sha(PROTOCOL),frozen_D0_core_reused=True,same_development8_and_seed42=True,expected_model_episodes=18,endpoint_gate='Training COMPLETE8000, finite/frozen/val0, overlay source checksum and keys verified at launch'))
 log('L_CPU_PREFLIGHT_PASS',protocol_sha256=sha(PROTOCOL),episodes=18)

def runtime_checks():
 protocol=json.loads(PROTOCOL.read_text())
 assert sha(SELECTION)==protocol['selection_sha256'] and sha(STATS)==protocol['stats_sha256']
 assert sha(D0/'protocol.json')==protocol['d0_protocol_sha256']
 for path,digest in protocol['source_sha256'].items():assert sha(path)==digest,('Source changed',path)
 assert json.loads((D0/'queue_status.json').read_text())['status']=='COMPLETE'
 summary_path=TRAINING/'arms/L/training_summary.json';summary=json.loads(summary_path.read_text())
 assert summary['status']=='COMPLETE' and summary['completed_updates']==8000 and summary['frozen_vlm_unchanged'] and summary['val_optimizer_updates']==0
 endpoint=protocol['checkpoint'];actual=sha(endpoint['path']);assert actual==summary['checkpoint_records']['8000']['checkpoint_sha256']
 endpoint['sha256']=actual
 endpoint_record=dict(**endpoint,training_summary_sha256=sha(summary_path),protocol_sha256=sha(PROTOCOL))
 helpers.immutable_json(ROOT/'endpoint_manifest.json',endpoint_record)
 for jobs in protocol['jobs'].values():
  for j in jobs:assert sha(j['scene']['annotation'])==j['scene']['annotation_sha256']
 assert json.loads((ROOT/'preflight.json').read_text())['protocol_sha256']==sha(PROTOCOL)
 return protocol

def completion(job):
 folder=ROOT/'rollouts'/job['job_id'];path=folder/'completed.json'
 if not path.exists():
  assert not folder.exists(),('Partial episode exists; retained for review',folder)
  return None
 data=json.loads(path.read_text());assert data['protocol_sha256']==sha(PROTOCOL)
 for name,digest in data['files_sha256'].items():assert sha(folder/name)==digest
 return json.loads((folder/'summary.json').read_text())

def compare_steps(current,previous,keys):
 counts={key:0 for key in keys};first={};max_errors={key:0. for key in keys}
 import numpy as np
 for index,(a,b) in enumerate(zip(current,previous),1):
  for key in keys:
   av,bv=a[key],b[key]
   if av!=bv:
    counts[key]+=1;first.setdefault(key,index)
    if isinstance(av,(list,float,int)) and isinstance(bv,(list,float,int)):
     try:max_errors[key]=max(max_errors[key],float(np.max(np.abs(np.asarray(av)-np.asarray(bv)))))
     except (TypeError,ValueError):pass
 from scipy.spatial.transform import Rotation
 n=min(len(current),len(previous));a=current[:n];b=previous[:n]
 position_max=max((float(np.linalg.norm(np.asarray(x['measured_position'])-y['measured_position'])) for x,y in zip(a,b)),default=0.)
 rotation_max=max((float((Rotation.from_quat(x['measured_quaternion'])*Rotation.from_quat(y['measured_quaternion']).inv()).magnitude()) for x,y in zip(a,b)),default=0.)
 width_max=max((abs(x['measured_width']-y['measured_width']) for x,y in zip(a,b)),default=0.)
 event_fields=dict(held_id=lambda r:r['grasp_after']['held_apple_id'],detached_count=lambda r:r['info']['apple_detached_count'],legacy_success=lambda r:r['info']['success'],ik_failed=lambda r:r['info']['ik_failed'],branch_break_count=lambda r:r['info']['branch_break_count'],advance=lambda r:r['advance'],dwell_steps=lambda r:r['dwell_steps'])
 events={key:all(fun(x)==fun(y) for x,y in zip(a,b)) for key,fun in event_fields.items()}
 targets_exact=all(all(x[k]==y[k] for k in ['target_position','target_rotation','target_width']) for x,y in zip(a,b))
 tolerance=dict(position_norm_m=1e-4,rotation_geodesic_rad=1e-4,width_m=1e-4)
 numerical=position_max<=1e-4 and rotation_max<=1e-4 and width_max<=1e-4
 return dict(compared_steps=n,current_steps=len(current),reference_steps=len(previous),field_mismatch_steps=counts,first_mismatch=first,max_numeric_abs_difference=max_errors,exact_prefix=not any(counts.values()),targets_exact=targets_exact,event_sequences_exact=events,physical_max_difference=dict(position_norm_m=position_max,rotation_geodesic_rad=rotation_max,width_m=width_max),gt_numerical_tolerances=tolerance,gt_numerical_control_parity=bool(targets_exact and numerical and all(events.values())),note='GT numerical tolerance is separate from raw bit equality; never used to declare model predicted targets equal')

def run(stage):
 import numpy as np
 import torch
 from checkpoint_io import load_trainable_overlay
 torch.set_num_threads(1)
 protocol=runtime_checks();jobs=protocol['jobs'][stage];completed={j['job_id']:completion(j) for j in jobs}
 if all(completed.values()):log('STAGE_ALREADY_COMPLETE',stage=stage);return
 lock=(ROOT/'gpu_eval.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 # Shared experimental GPU guard also prevents collision with old A/B training.
 shared=(AB/'.gpu_training.lock').open('r');fcntl.flock(shared.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 experiment_lock=(bootstrap.EXPERIMENT/'.gpu.lock').open('a');fcntl.flock(experiment_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 loop=module(D0/'reach_loop.py','grasp_runtime_loop');loop.ROOT=ROOT;loop.load_runtime()
 analyze=module(XR0/'diagnosis_1003/analyze_rollouts.py','grasp_trace_analyzer')
 passive=module(OLD/'passive_metrics.py','grasp_passive_metrics')
 active={};base_save,base_line,base_snapshot=loop.save,loop.line,loop.snapshot_grasp
 with helpers.Heartbeat(stage,len(jobs)) as heartbeat:
  def tags():return dict(stage=stage,cohort=active['cohort'],inference_seed=active['inference_seed'],actual_inference_seed=active['inference_seed'],job_id=active['job_id'],evaluation_protocol_sha256=sha(PROTOCOL))
  def save(path,data):
   if path.name in ['initial.json','summary.json']:data.update(tags())
   base_save(path,data)
  def line(stream,data):
   data.update(tags());base_line(stream,data)
   if 'control_step' in data:heartbeat.update(control_step=data['control_step'],chunk_id=data['chunk_id'],target_k=data['target_k'])
  def snapshot(env):
   snap=base_snapshot(env);apple_id=active['planned_apple_id'];body=int(env.tm.apple_bodies[apple_id])
   snap['planned_apple']=passive.planned_apple_observation(snap,apple_id,body,env.sim.body_q_np()[body,:3]);return snap
  loop.save,loop.line,loop.snapshot_grasp=save,line,snapshot
  policy=None;checkpoint='per-scene absolute-world GT target stream'
  if not jobs[0]['is_gt']:
   from mibot.server.orchard_policy import OrchardPolicy
   log('MODEL_LOADING',stage=stage);policy=OrchardPolicy(str(BASE),str(PROCESSOR),str(STATS))
   checkpoint=protocol['checkpoint']['path'];loaded=load_trainable_overlay(policy.model,checkpoint,base_checkpoint=BASE)
   assert loaded['checkpoint_sha256']==protocol['checkpoint']['sha256'] and loaded['step']==8000
   dtypes={g:dict(Counter(str(p.dtype) for n,p in policy.model.named_parameters() if n.startswith('vlm.')==isvlm)) for g,isvlm in [('vlm',True),('non_vlm',False)]}
   assert dtypes=={'vlm':{'torch.bfloat16':713},'non_vlm':{'torch.float32':219}}
   helpers.immutable_json(ROOT/f'load_manifest_{stage}.json',dict(load_report=loaded,dtypes=dtypes,protocol_sha256=sha(PROTOCOL)))
  results=[]
  for j in jobs:
   if completed[j['job_id']] is not None:results.append(completed[j['job_id']]);continue
   active.clear();active.update(j);traj=json.loads(Path(j['scene']['annotation']).read_text())
   active['planned_apple_id']=int(traj['orchardbench']['fixed_base_expert']['selected_apple_debug_index'])
   bound=None if j['is_gt'] else helpers.BoundSeedPolicy(policy,j['inference_seed'])
   heartbeat.update(status='EPISODE',job_id=j['job_id'],control_step=0,stage_completed=len(results))
   log('EPISODE_START',job_id=j['job_id'],processed_targets=j['processed_targets'])
   result=loop.episode(bound,j['provider'],j['scene'],checkpoint,j['video'],protocol['selection_sha256'],processed_targets=j['processed_targets'],is_gt=j['is_gt'])
   folder=ROOT/'rollouts'/j['job_id'];steps=read_lines(folder/'steps.jsonl');chunks=read_lines(folder/'chunks.jsonl')
   assert result['termination']!='error',('Episode error retained',folder)
   if bound is not None:assert len(bound.calls)==result['replans']==len(chunks)
   save_new(folder/'seed_delivery.json',dict(declared=j['inference_seed'],calls=[] if bound is None else bound.calls))
   analysis_summary=dict(result,checkpoint='gt') if j['is_gt'] else result
   metrics=analyze.analyze_episode(analysis_summary,steps,chunks,folder);metrics['provider']=result['checkpoint'];metrics['gt_analyzer_alias_applied']=bool(j['is_gt']);metrics.update(helpers.held_runs(steps));metrics.update(tags())
   gtref=OLD/'rollouts'/f'gt_reach-conditioned_{j["scene"]["seed"]}'/'chunks.jsonl'
   metrics.update(passive.diagnostics(steps,chunks,read_lines(gtref)))
   metrics.update(strict_placement=result['strict_placement'],strict_success=result['strict_success'],legacy_success_ever=result['legacy_success_ever'],processed_targets_per_replan=j['processed_targets'])
   # Runtime reimplementation parity: original success still equals exact fruit geometry.
   assert all(len(s['fruit_state']['in_bucket_apple_ids'])==s['info']['apple_in_bucket_count'] for s in steps)
   assert result['scene_check_pass'] and metrics['observer_consistency']['mismatch_fields_total']==0 and metrics['summary_consistency']['mismatch_count']==0
   assert all(all(s['observer_consistency'].values()) for s in steps)
   baseline=D0/'rollouts'/f'B8000_targets{j["processed_targets"]}_rng{j["inference_seed"]}_reach-conditioned_{j["scene"]["seed"]}'
   comparison=compare_steps(steps,read_lines(baseline/'steps.jsonl'),['target_position','target_rotation','target_width','measured_position','measured_quaternion','measured_width','requested_command','advance','dwell_steps'])
   comparison['reference']=str(baseline);comparison['role']='Learned policy L versus matching current B baseline; differences are expected and are not control parity failures'
   current_initial=json.loads((folder/'initial.json').read_text())['obs'];baseline_initial=json.loads((baseline/'initial.json').read_text())['obs']
   assert current_initial==baseline_initial,'Current L/B paired resets must be exact'
   metrics['paired_baseline_reset_exact']=True
   metrics['control_regression']=comparison
   if comparison is not None:save_new(folder/'control_regression.json',comparison)
   save_new(folder/'metrics.json',metrics)
   # Differences are preserved and surfaced; no case exclusion or silent retry.
   if comparison is not None:log('L_BASELINE_COMPARISON',job_id=j['job_id'],baseline=str(baseline),resets_exact=True)
   files=['summary.json','steps.jsonl','chunks.jsonl','initial.json','scene_check.json','seed_delivery.json','metrics.json']
   atomic_json(folder/'completed.json',dict(status='COMPLETE',job_id=j['job_id'],protocol_sha256=sha(PROTOCOL),checkpoint_sha256=None if j['is_gt'] else protocol['checkpoint']['sha256'],files_sha256={name:sha(folder/name) for name in files}))
   results.append(result);heartbeat.update(stage_completed=len(results))
   log('EPISODE_COMPLETE',job_id=j['job_id'],held15=metrics['same_fruit_held_at_least_15_steps'],strict_success=result['strict_success'],legacy_success_ever=result['legacy_success_ever'],steps=result['control_steps'],replans=result['replans'])
  runtime_checks();policy=None;gc.collect();torch.cuda.empty_cache()
  atomic_json(ROOT/f'stage_{stage}_complete.json',dict(status='COMPLETE',stage=stage,episodes=len(results),strict_success=sum(r['strict_success'] for r in results),legacy_success_ever=sum(r['legacy_success_ever'] for r in results),protocol_sha256=sha(PROTOCOL)))
  heartbeat.update(status='COMPLETE',stage_completed=len(results),job_id=None)
 experiment_lock.close();shared.close();lock.close()

def queue():
 protocol=runtime_checks();lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 for stage in protocol['queue_order']:
  atomic_json(ROOT/'queue_status.json',dict(status='RUNNING',stage=stage,pid=__import__('os').getpid(),updated_unix=time.time()))
  with (ROOT/f'console_{stage}.log').open('a') as output:
   process=subprocess.run([sys.executable,'-B','-u',str(Path(__file__)),'--run',stage],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
  if process.returncode:
   atomic_json(ROOT/'queue_status.json',dict(status='ERROR',stage=stage,returncode=process.returncode,updated_unix=time.time()));raise RuntimeError('Evaluation failed; retained for review: '+stage)
 atomic_json(ROOT/'queue_status.json',dict(status='COMPLETE',episodes=18,updated_unix=time.time(),protocol_sha256=sha(PROTOCOL)))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',choices=STAGES);parser.add_argument('--queue',action='store_true');args=parser.parse_args()
 if args.prepare:prepare()
 elif args.run:run(args.run)
 elif args.queue:queue()
 else:parser.error('Choose --prepare, --run, or --queue')
