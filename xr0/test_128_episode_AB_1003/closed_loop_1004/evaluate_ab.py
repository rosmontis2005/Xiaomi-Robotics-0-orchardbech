"""Predeclared A/B step8000 evaluation; reuses the unchanged M0 reach loop.

--prepare is CPU-only and does not import the simulator or initialize CUDA.
--run GT/A/B starts only after both training arms completed, under root GPU grant.
All task metrics are passive; no goal-hold gate, phase switch or GT model input.
"""
import bootstrap
import argparse
import ast
from collections import Counter
import gc
import hashlib
import importlib.util
import json
import os
import threading
import subprocess
import sys
import fcntl
from pathlib import Path
import time
import traceback

ROOT,AB,XR0,ORCHARD=bootstrap.ROOT,bootstrap.AB,bootstrap.XR0,bootstrap.ORCHARD
M0=XR0/'test_small_set_fit_1003'
SELECTION=AB/'selection.json'
BASE=XR0/'outputs/orchard_v1_2650/project_orchardbench/orchard_v1_2650_frozen_vlm_seed42/epoch=0-step=10000.ckpt'
STATS=ORCHARD/'data/orchard_v1_2650/filtered/action_stats.json'
PROCESSOR=XR0.parent/'checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D'
PROTOCOL=ROOT/'protocol.json'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def save_new(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')

def log(event,**data):
 row=dict(unix=time.time(),event=event,**data)
 with (ROOT/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)

def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def scene_list():
 selection=json.loads(SELECTION.read_text())
 dev=selection['development_scenes'];held=selection['heldout_scenes']
 assert len(dev)==8 and len(held)==12 and dev==selection['closed_loop_scenes']
 assert len({s['seed'] for s in dev+held})==20
 assert all(s['split']=='val' for s in held)
 assert not {s['episode_id'] for s in held}&set(selection['train_episode_ids'])
 return selection,[dict(s,cohort='dev') for s in dev]+[dict(s,cohort='heldout') for s in held]

def jobs_for(stage,scenes):
 jobs=[]
 for scene in scenes:
  seeds=[None] if stage=='GT' else [42] if scene['cohort']=='dev' else [42,43]
  for seed in seeds:
   provider='gt' if stage=='GT' else f'{stage}_step8000_rng{seed}'
   video=(stage!='GT' and seed==42 and scene['cohort']=='dev' and scene['seed'] in [s['seed'] for s in scenes[:6]])
   jobs.append(dict(stage=stage,arm=None if stage=='GT' else stage,provider=provider,scene=scene,inference_seed=seed,video=video,checkpoint_step=None if stage=='GT' else 8000,job_id=f'{provider}_reach-conditioned_{scene["seed"]}'))
 return jobs

class BoundSeedPolicy:
 """The reused loop asks for seed42; explicitly bind the declared experiment seed."""
 def __init__(self,policy,seed):self.policy=policy;self.seed=int(seed);self.calls=[]
 @property
 def adapter(self):return self.policy.adapter
 def predict(self,obs,seed=42):
  self.calls.append(dict(replan_index=len(self.calls),legacy_requested_seed=int(seed),actual_inference_seed=self.seed))
  return self.policy.predict(obs,seed=self.seed)

def held_runs(steps):
 runs=[];current=None;start=None;previous_step=None
 for row in steps:
  held=row['grasp_after']['held_apple_id'];step=row['control_step']
  if held!=current or (previous_step is not None and step!=previous_step+1):
   if current is not None:runs.append(dict(apple_id=current,start_step=start,end_step=previous_step,consecutive_control_steps=previous_step-start+1))
   current=held;start=step if held is not None else None
  previous_step=step
 if current is not None:runs.append(dict(apple_id=current,start_step=start,end_step=previous_step,consecutive_control_steps=previous_step-start+1))
 sustained=[r for r in runs if r['consecutive_control_steps']>=15]
 return dict(held_runs=runs,same_fruit_held_at_least_15_steps=bool(sustained),first_held15_qualification_step=sustained[0]['start_step']+14 if sustained else None,first_sustained_grasp_start_step=sustained[0]['start_step'] if sustained else None,max_same_fruit_held_steps=max([r['consecutive_control_steps'] for r in runs],default=0),sustained_grasp_started_by_step300=any(r['start_step']<=300 for r in sustained),held15_qualified_by_step300=any(r['start_step']+14<=300 for r in sustained),metric_only_no_control_gate=True)

def pinned_sources():
 expected=json.loads((AB/'arms/B/run_manifest.json').read_text())['provenance']['source_sha256']
 for path,digest in expected.items():assert sha(path)==digest,('Frozen training source changed',path)
 extra=[Path(__file__),ROOT/'bootstrap.py',ROOT/'passive_metrics.py',M0/'evaluate_reach.py',XR0/'diagnosis_1003/observers.py',XR0/'diagnosis_1003/analyze_rollouts.py',XR0/'tools/eval_orchard_learning_curve.py',ORCHARD/'log/v1_2650_collection_replay_filter/sources/frozen_gate1_oracle.py']
 return {**expected,**{str(p):sha(p) for p in extra}}

def prepare():
 import torch
 assert not torch.cuda.is_initialized()
 selection,scenes=scene_list();loop=module(M0/'evaluate_reach.py','ab_cpu_m0_loop')
 assert loop.advance_decision(.01,.08,1)==(True,False,True,'reached')
 assert loop.advance_decision(.011,.08,29)==(False,False,False,'hold')
 assert loop.advance_decision(.011,.08,30)==(False,True,True,'maximum_dwell')
 assert loop.advance_decision(.01,.081,30)==(False,True,True,'maximum_dwell')
 class FakePolicy:
  adapter='sentinel'
  def __init__(self):self.seeds=[]
  def predict(self,obs,seed):self.seeds.append(seed);return seed
 for seed in [42,43]:
  fake=FakePolicy();wrapped=BoundSeedPolicy(fake,seed)
  assert wrapped.predict({},seed=42)==seed and wrapped.predict({},seed=42)==seed
  assert fake.seeds==[seed,seed] and wrapped.adapter=='sentinel'
  assert all(c['actual_inference_seed']==seed for c in wrapped.calls)
 def fake_rows(ids):return [dict(control_step=i+1,grasp_after=dict(held_apple_id=held)) for i,held in enumerate(ids)]
 assert not held_runs(fake_rows([1]*14+[None]+[1]*14))['same_fruit_held_at_least_15_steps']
 assert not held_runs(fake_rows([1]*8+[2]*8))['same_fruit_held_at_least_15_steps']
 assert held_runs(fake_rows([None]+[7]*15))['first_held15_qualification_step']==16
 checks=[]
 from decord import VideoReader
 for scene in scenes:
  path=Path(scene['annotation']);assert sha(path)==scene['annotation_sha256'];data=json.loads(path.read_text())
  assert data['seed']==scene['seed'] and data['episode_id']==scene['episode_id'] and data['split']==scene['split']
  n=data['num_frames'];assert n>=30 and data['record_fps']==30 and data['orchardbench']['timestamps'][0]==0
  consumed=[]
  for first in range(0,n,30):
   anchor=min(first,n-30);consumed.extend(anchor+k for k in range(first-anchor,min(first+30,n)-anchor))
  assert consumed==list(range(n))
  views=[]
  for key in ['ego','wrist_left']:
   video=Path(data['observations'][key][0]['path']);reader=VideoReader(str(video),num_threads=1)
   assert len(reader)==n and abs(reader.get_avg_fps()-30)<1e-3
   frame=reader[0].asnumpy();assert frame.shape==(144,192,3)
   views.append(dict(view=key,path=str(video),frame_count=len(reader),first_decoded_frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest()))
  checks.append(dict(seed=scene['seed'],episode_id=scene['episode_id'],cohort=scene['cohort'],split=scene['split'],annotation_sha256=sha(path),num_frames=n,gt_targets_consumed_exactly_once=True,source_video_checks=views))
 expected_sha='671d16cb542bf509978cf016831e7b632815a49542ef108a0a171d334fddc6d9'
 assert sha(XR0/'mibot/utils/orchard_checkpoint.py')==expected_sha
 assert json.loads((AB/'precision_fix.json').read_text())['status']=='PASS'
 jobs={stage:jobs_for(stage,scenes) for stage in ['GT','A','B']}
 assert [len(jobs[s]) for s in ['GT','A','B']]==[20,32,32]
 assert all(jobs['A'][i]['scene']['seed']==jobs['B'][i]['scene']['seed'] and jobs['A'][i]['inference_seed']==jobs['B'][i]['inference_seed'] for i in range(32))
 protocol=dict(schema='orchard_ab_closed_loop_v1',created_unix=time.time(),primary_endpoint_step=8000,endpoint_selection='Predeclared step8000 for both arms; no best checkpoint or outcome selection',selection_path=str(SELECTION),selection_sha256=sha(SELECTION),stats_path=str(STATS),stats_sha256=sha(STATS),precision_fix_sha256=sha(AB/'precision_fix.json'),source_sha256=pinned_sources(),reach_protocol=dict(position_tolerance_m=.01,rotation_tolerance_rad=.08,max_dwell_control_steps=30,control_budget=900,targets_per_chunk=30,fixed_chunk_anchor=True,grasp_mode='benchmark_assist',new_control_gates=False),gt_policy='Run all20 selected scenes through the same frozen student adapter/controller; retain every outcome even if current GT fails',queue_order=['GT','A','B'],jobs=jobs,model_episode_count=64,gt_episode_count=20,inference_seed_rule='Same declared seed reset at every replan, following production OrchardPolicy semantics; seeds42/43 explicitly bound and logged',video_rule='Each arm seed42: first6 development scenes (old4 + new2); all episodes retain complete steps/chunks',checkpoint_paths={arm:str(AB/'arms'/arm/'checkpoints/step_8000_trainable.pt') for arm in ['A','B']},diagnostic_only_held15=True,secondary_train_best=dict(automatic=False,primary_queue_must_complete=True,cohort='dev',inference_seeds=[42],rule='Each arm own previously selected train-only best, only if different from8000; separate results and explicit root GPU decision'),limits=['Held15 counts consecutive post-control boundary samples for the same fruit while normal control continues; no pose hold is imposed.','Nearest-fruit distance after grasp is not reach evidence; pre-first-grasp distances preserve fruit ID.','Curated fixed scenes and two inference seeds are a development comparison, not a natural population success-rate estimate.','GT identity/phase/trajectory never enter model input or online model decisions.','Current GT task failure is retained and flagged, not used to exclude or replace a scene.'])
 save_new(PROTOCOL,protocol)
 save_new(ROOT/'protocol_addendum.json',dict(reason='Re-run all20 GT: old4 GT full evaluator file SHA predates later precision-probe additions, so strict identical-file reuse condition cannot be established from retained source artifacts.',old_gt_manifest=str(M0/'evaluation_manifest_gt.json'),old_gt_manifest_sha256=sha(M0/'evaluation_manifest_gt.json'),decision='No old GT episodes are reused in current20-scene outcome totals; old4 remain historical context',root_approved=True,production_or_training_sources_modified=False))
 analyzer=module(XR0/'diagnosis_1003/analyze_rollouts.py','ab_cpu_trace_analysis')
 fixture=M0/'rollouts/best_reach-conditioned_2014461'
 fixture_steps=analyzer.read_jsonl(fixture/'steps.jsonl');fixture_chunks=analyzer.read_jsonl(fixture/'chunks.jsonl')
 fixture_summary=json.loads((fixture/'summary.json').read_text())
 analyzed=analyzer.analyze_episode(fixture_summary,fixture_steps,fixture_chunks,fixture)
 assert analyzed['observer_consistency']['mismatch_fields_total']==0
 assert analyzed['summary_consistency']['mismatch_count']==0
 assert all(all(row['observer_consistency'].values()) for row in fixture_steps)
 import passive_metrics
 gt_chunks=analyzer.read_jsonl(M0/'rollouts/gt_reach-conditioned_2014461'/'chunks.jsonl')
 passive=passive_metrics.diagnostics(fixture_steps,fixture_chunks,gt_chunks)
 assert passive['first_chunk_teacher_error']['full30']['position_error_m']['count']==30
 assert passive['minimum_tcp_to_nearest_apple_before_first_grasp']['apple']['apple_id']==28
 base=dict(hand_position_world=[0,0,0],hand_quaternion_world=[0,0,0,1],tcp_position_world=[0,0,.1034],contact_apples=[dict(apple_id=1,touching_finger_bodies=[10,11])],expected_finger_bodies=[10,11])
 assert passive_metrics.planned_apple_observation(base,1,99,[0,0,.095])['geometry_eligible']
 assert not passive_metrics.planned_apple_observation(base,1,99,[0,0,.125])['geometry_eligible']
 assert not torch.cuda.is_initialized()
 report=dict(status='PASS',cuda_initialized=False,simulator_imported=False,scenes=checks,seed_binding_cpu_test='PASS for42/43 across repeated calls',held_streak_cpu_test='PASS for14+break+14, fruit switch8+8, continuous15',reach_threshold_and_dwell_tests='PASS',tail_window_coverage='PASS all20',protocol_sha256=sha(PROTOCOL),existing_900step_analyzer_fixture='PASS; no observer/summary mismatch',planned_geometry_cpu_test='PASS strict palm boundary',endpoint_paths_predeclared=True,endpoint_files_verified_at_runtime_after_both_arms_complete=True)
 save_new(ROOT/'preflight.json',report);log('CPU_PREFLIGHT_PASS',scenes=20,gt_episodes=20,model_episodes=64,protocol_sha256=sha(PROTOCOL))

def runtime_checks():
 import torch
 protocol=json.loads(PROTOCOL.read_text());assert sha(SELECTION)==protocol['selection_sha256'] and sha(STATS)==protocol['stats_sha256']
 for path,digest in protocol['source_sha256'].items():assert sha(path)==digest,('Pinned source changed',path)
 records={}
 for arm in ['A','B']:
  summary=json.loads((AB/'arms'/arm/'training_summary.json').read_text())
  assert summary['status']=='COMPLETE' and summary['completed_updates']==8000 and summary['frozen_vlm_unchanged'] and summary['val_optimizer_updates']==0
  path=Path(protocol['checkpoint_paths'][arm]);payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
  assert payload['step']==8000 and payload['format']=='orchard_m0_trainable_v1'
  provenance=payload['metadata']['provenance'];assert provenance['arm']==arm and provenance['selection_sha256']==protocol['selection_sha256']
  assert provenance['source_sha256'][str(XR0/'mibot/utils/orchard_checkpoint.py')]==protocol['source_sha256'][str(XR0/'mibot/utils/orchard_checkpoint.py')]
  assert Path(payload['base_checkpoint']).resolve()==BASE.resolve()
  digest=sha(path);assert digest==summary['checkpoint_records']['8000']['checkpoint_sha256']
  records[arm]=dict(path=str(path),step=8000,sha256=digest,base_checkpoint=str(BASE),base_sha256=payload['base_sha256'],training_summary_sha256=sha(AB/'arms'/arm/'training_summary.json'))
  del payload
 pinned=ROOT/'endpoint_checkpoints.json'
 if pinned.exists():assert json.loads(pinned.read_text())==records
 else:save_new(pinned,records)
 return protocol,records

def enrich_result(result,steps,job):
 held=held_runs(steps);infoevents=[s for s in steps if s['info']['grasp_assist_triggered'] or s['grasp_after']['held_apple_id'] is not None]
 first_grasp=infoevents[0]['control_step'] if infoevents else None
 first_detach=next((s['control_step'] for s in steps if s['info']['apple_detached_count']>0),None)
 actual_grasp=bool(infoevents);assert actual_grasp==result['ever_grasped']
 return dict(**held,first_grasp_control_step=first_grasp,first_detach_control_step=first_detach,grasp_by_step300=first_grasp is not None and first_grasp<=300,actual_grasp_event=actual_grasp,grasp_event_definition='grasp_assist_triggered or actual held ID; never target width proxy')

def atomic_json(path,data):
 temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
 with temporary.open('w') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temporary,path)

def immutable_json(path,data):
 if path.exists():assert json.loads(path.read_text())==data,('Existing immutable record differs',str(path))
 else:save_new(path,data)

class Heartbeat:
 def __init__(self,label,total):
  self.state=dict(stage=label,status='STARTING',stage_total=total,stage_completed=0,control_step=None,job_id=None)
  self.lock=threading.Lock();self.stop_event=threading.Event();self.thread=threading.Thread(target=self.loop,daemon=True)
 def write(self):
  with self.lock:
   row=dict(self.state,heartbeat_unix=time.time(),pid=os.getpid(),protocol_sha256=sha(PROTOCOL))
   row['all_completed_records']=len(list((ROOT/'rollouts').glob('*/completed.json')))
  atomic_json(ROOT/f'heartbeat_{self.state["stage"]}.json',row)
 def loop(self):
  while not self.stop_event.wait(10):self.write()
 def update(self,**items):
  with self.lock:self.state.update(items)
 def __enter__(self):self.write();self.thread.start();return self
 def __exit__(self,kind,value,tb):
  self.stop_event.set();self.thread.join()
  if kind:self.update(status='ERROR',error=str(value))
  self.write()

def completion_for(job,checkpoint_sha):
 directory=ROOT/'rollouts'/job['job_id'];record=directory/'completed.json'
 if not directory.exists():return None
 if not record.exists():raise RuntimeError('Partial/failed episode retained for explicit review: '+str(directory))
 done=json.loads(record.read_text())
 assert done['job_id']==job['job_id'] and done['protocol_sha256']==sha(PROTOCOL)
 assert done['inference_seed']==job['inference_seed'] and done['checkpoint_sha256']==checkpoint_sha
 assert done['annotation_sha256']==job['scene']['annotation_sha256']
 for name,digest in done['files_sha256'].items():assert sha(directory/name)==digest,('Completed artifact changed',directory/name)
 return json.loads((directory/'summary.json').read_text())

def secondary_setup(arm,protocol):
 import torch
 for stage in ['GT','A','B']:assert json.loads((ROOT/f'stage_{stage}_complete.json').read_text())['status']=='COMPLETE'
 record=json.loads((AB/'arms'/arm/'best_checkpoint.json').read_text())
 assert record['rule']['criterion']=='lexicographic, selected TRAIN panel only; VAL is report-only'
 step=int(record['step'])
 if step==8000:return None,[]
 path=AB/'arms'/arm/'checkpoints'/f'step_{step:04d}_trainable.pt'
 assert path.resolve()==Path(record['path']).resolve() and sha(path)==record['checkpoint_sha256']
 payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
 assert payload['step']==step and payload['metadata']['provenance']['arm']==arm
 assert payload['metadata']['provenance']['selection_sha256']==protocol['selection_sha256']
 endpoint=dict(path=str(path),step=step,sha256=record['checkpoint_sha256'],base_checkpoint=str(BASE),base_sha256=payload['base_sha256'],train_best_record_sha256=sha(AB/'arms'/arm/'best_checkpoint.json'))
 jobs=[]
 for original in protocol['jobs'][arm]:
  if original['scene']['cohort']!='dev':continue
  job=dict(original);job['checkpoint_step']=step;job['provider']=f'{arm}_trainbest_step{step}_rng42';job['job_id']=f'{job["provider"]}_reach-conditioned_{job["scene"]["seed"]}';job['comparison_role']='secondary_train_best'
  jobs.append(job)
 assert len(jobs)==8
 immutable_json(ROOT/f'secondary_endpoint_{arm}.json',endpoint)
 return endpoint,jobs

def run(stage,secondary=False):
 import numpy as np
 import torch
 from checkpoint_io import load_trainable_overlay
 import passive_metrics
 torch.set_num_threads(1)
 protocol,endpoints=runtime_checks();jobs=protocol['jobs'][stage]
 label=stage;endpoint=None if stage=='GT' else endpoints[stage]
 if secondary:
  assert stage in ['A','B'];endpoint,jobs=secondary_setup(stage,protocol);label=stage+'_trainbest'
  if endpoint is None:log('SECONDARY_NOT_NEEDED',arm=stage,reason='train-only best already equals primary8000');return
 for job in jobs:job.setdefault('comparison_role','primary_endpoint' if stage!='GT' else 'current_gt_control')
 checkpoint_sha=None if stage=='GT' else endpoint['sha256']
 complete_path=ROOT/f'stage_{label}_complete.json'
 if complete_path.exists():
  done=json.loads(complete_path.read_text());assert done['status']=='COMPLETE' and done['protocol_sha256']==sha(PROTOCOL)
  assert all(completion_for(job,checkpoint_sha) is not None for job in jobs)
  log('STAGE_ALREADY_COMPLETE',stage=label,episodes=len(jobs));return
 completed={job['job_id']:completion_for(job,checkpoint_sha) for job in jobs}
 gpu_lock=(ROOT/'gpu_eval.lock').open('a')
 fcntl.flock(gpu_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 with Heartbeat(label,len(jobs)) as heartbeat:
  heartbeat.update(status='INITIALIZING',stage_completed=sum(r is not None for r in completed.values()))
  loop=module(M0/'evaluate_reach.py','ab_frozen_m0_runtime');loop.ROOT=ROOT;loop.load_runtime()
  assert loop.CONFIG.max_control_steps==900 and loop.CONFIG.grasp_mode=='benchmark_assist'
  base_save,base_line,base_check,base_snapshot=loop.save,loop.line,loop.check_reset_scene,loop.snapshot_grasp
  analyze=module(XR0/'diagnosis_1003/analyze_rollouts.py','ab_readonly_trace_analysis')
  active={}
  def tags():
   return dict(arm=active['arm'],cohort=active['scene']['cohort'],actual_inference_seed=active['inference_seed'],inference_seed=active['inference_seed'],checkpoint_step=active['checkpoint_step'],job_id=active['job_id'],comparison_role=active['comparison_role'])
  def tagged_save(path,data):
   if path.name in ['initial.json','summary.json']:
    data.update(**tags(),evaluation_protocol_sha256=sha(PROTOCOL),source=str(Path(__file__)),reused_control_loop_source=str(M0/'evaluate_reach.py'))
   base_save(path,data)
  def tagged_line(stream,data):
   data.update(tags());base_line(stream,data)
   if 'control_step' in data:heartbeat.update(control_step=data['control_step'],chunk_id=data['chunk_id'],target_k=data['target_k'])
  def passive_snapshot(env):
   snapshot=base_snapshot(env)
   # Privileged ID is used only here to log a live fruit pose, never by policy or scheduler.
   apple_id=active['planned_apple_id'];body=int(env.tm.apple_bodies[apple_id])
   snapshot['planned_apple']=passive_metrics.planned_apple_observation(snapshot,apple_id,body,env.sim.body_q_np()[body,:3])
   return snapshot
  def checked_reset(obs,info,traj,directory,initial):
   report=base_check(obs,info,traj,directory,initial)
   if active['stage']!='GT':
    gt=ROOT/'rollouts'/f'gt_reach-conditioned_{traj["seed"]}'
    previous=json.loads((gt/'initial.json').read_text());summary=json.loads((gt/'summary.json').read_text())
    checks={k:(value==previous['obs'][k] if k.startswith('rgb_') else bool(np.array_equal(np.asarray(value),np.asarray(previous['obs'][k])))) for k,value in initial['obs'].items()}
    report['current_gt_pairing']=dict(path=str(gt),exact_match=all(checks.values()),checks=checks,gt_task_success=summary['success'],gt_runtime_valid=summary['termination']!='error')
    report['state_and_identity_pass']=report['state_and_identity_pass'] and all(checks.values()) and summary['termination']!='error'
   return report
  loop.save,loop.line,loop.check_reset_scene,loop.snapshot_grasp=tagged_save,tagged_line,checked_reset,passive_snapshot
  policy=None;checkpoint='per-scene absolute-world GT targets'
  if stage!='GT':
   assert all(completion_for(job,None) is not None for job in protocol['jobs']['GT'])
   from mibot.server.orchard_policy import OrchardPolicy
   log('MODEL_LOADING',arm=stage,comparison_role='secondary_train_best' if secondary else 'primary_endpoint')
   policy=OrchardPolicy(str(BASE),str(PROCESSOR),str(STATS));checkpoint=endpoint['path']
   load_report=load_trainable_overlay(policy.model,checkpoint,base_checkpoint=BASE)
   assert load_report['step']==endpoint['step'] and load_report['checkpoint_sha256']==endpoint['sha256']
   dtypes={group:dict(Counter(str(p.dtype) for n,p in policy.model.named_parameters() if n.startswith('vlm.')==isvlm)) for group,isvlm in [('vlm',True),('non_vlm',False)]}
   assert dtypes=={'vlm':{'torch.bfloat16':713},'non_vlm':{'torch.float32':219}}
   immutable_json(ROOT/f'load_manifest_{label}.json',dict(arm=stage,endpoint=endpoint,load_report=load_report,parameter_dtypes=dtypes,protocol_sha256=sha(PROTOCOL)))
  results=[];bound=None
  for job in jobs:
   active.clear();active.update(job);directory=ROOT/'rollouts'/job['job_id']
   if completed[job['job_id']] is not None:
    results.append(completed[job['job_id']]);log('EXISTING_COMPLETED_EPISODE',job_id=job['job_id']);continue
   annotation=json.loads(Path(job['scene']['annotation']).read_text())
   active['planned_apple_id']=int(annotation['orchardbench']['fixed_base_expert']['selected_apple_debug_index'])
   bound=None if stage=='GT' else BoundSeedPolicy(policy,job['inference_seed'])
   heartbeat.update(status='EPISODE',job_id=job['job_id'],control_step=0,stage_completed=len(results),scene_seed=job['scene']['seed'],inference_seed=job['inference_seed'])
   log('EPISODE_START',job_id=job['job_id'],arm=job['arm'],cohort=job['scene']['cohort'],inference_seed=job['inference_seed'])
   result=loop.episode(bound,job['provider'],job['scene'],checkpoint,job['video'],protocol['selection_sha256'])
   steps=analyze.read_jsonl(directory/'steps.jsonl');chunks=analyze.read_jsonl(directory/'chunks.jsonl')
   if bound is not None:
    assert len(bound.calls)==result['replans']==len(chunks)
    assert all(c['actual_inference_seed']==job['inference_seed'] for c in bound.calls)
    save_new(directory/'seed_delivery.json',dict(declared_inference_seed=job['inference_seed'],calls=bound.calls,all_calls_match=True))
   else:save_new(directory/'seed_delivery.json',dict(declared_inference_seed=None,model_calls=0,provider='absolute-world GT'))
   metrics=analyze.analyze_episode(result,steps,chunks,directory)
   metrics.update(enrich_result(result,steps,job));metrics.update(tags())
   gtchunks=chunks if stage=='GT' else analyze.read_jsonl(ROOT/'rollouts'/f'gt_reach-conditioned_{job["scene"]["seed"]}'/'chunks.jsonl')
   metrics.update(passive_metrics.diagnostics(steps,chunks,gtchunks))
   metrics['any_detach']=result['max_apple_detached_count']>0
   metrics['grasp_and_any_detach']=result['ever_grasped'] and result['max_apple_detached_count']>0
   metrics['same_grasped_fruit_detach_verified']=None
   metrics['detach_identity_note']='Environment exposes aggregate detached count, not the detached fruit ID; conjunction does not prove identical fruit.'
   first_success=next((r for r in steps if r['info']['success']),None)
   success_step=first_success['control_step'] if first_success else None
   metrics['held_apple_id_at_first_success']=first_success['grasp_after']['held_apple_id'] if first_success else None
   metrics['release_observed_strictly_before_success']=None if first_success is None else any(e['control_step']<success_step for e in metrics['observed_held_to_none_release_events'])
   metrics['release_observed_by_success_boundary']=None if first_success is None else any(e['control_step']<=success_step for e in metrics['observed_held_to_none_release_events'])
   metrics['benchmark_success_definition']=steps[-1]['info'].get('success_definition') if steps else None
   save_new(directory/'metrics.json',metrics)
   okay=(result['termination']!='error' and result['scene_check_pass'] and metrics['observer_consistency']['mismatch_fields_total']==0 and metrics['summary_consistency']['mismatch_count']==0 and all(all(r['observer_consistency'].values()) for r in steps))
   if not okay:
    atomic_json(directory/'failed.json',dict(job_id=job['job_id'],protocol_sha256=sha(PROTOCOL),termination=result['termination'],scene_check_pass=result['scene_check_pass'],observer=metrics['observer_consistency'],summary_consistency=metrics['summary_consistency']))
    raise RuntimeError('Episode runtime/reset/consistency error retained; stop for explicit review: '+str(directory))
   files=['summary.json','steps.jsonl','chunks.jsonl','initial.json','scene_check.json','seed_delivery.json','metrics.json']
   atomic_json(directory/'completed.json',dict(status='COMPLETE',job_id=job['job_id'],protocol_sha256=sha(PROTOCOL),inference_seed=job['inference_seed'],checkpoint_sha256=checkpoint_sha,annotation_sha256=job['scene']['annotation_sha256'],files_sha256={n:sha(directory/n) for n in files},success=result['success'],grasp=result['ever_grasped'],held15=metrics['same_fruit_held_at_least_15_steps'],completed_unix=time.time()))
   log('EPISODE_COMPLETE',job_id=job['job_id'],valid=True,grasp=result['ever_grasped'],held15=metrics['same_fruit_held_at_least_15_steps'],any_detach=result['max_apple_detached_count'],success=result['success'],steps=result['control_steps'])
   results.append(result);heartbeat.update(stage_completed=len(results))
  bound=None;policy=None;gc.collect();torch.cuda.empty_cache()
  for path,digest in protocol['source_sha256'].items():assert sha(path)==digest
  atomic_json(complete_path,dict(stage=label,status='COMPLETE',episodes=len(results),valid=len(results),grasp=sum(r['ever_grasped'] for r in results),any_detach=sum(r['max_apple_detached_count']>0 for r in results),grasp_and_any_detach=sum(r['ever_grasped'] and r['max_apple_detached_count']>0 for r in results),success=sum(r['success'] for r in results),protocol_sha256=sha(PROTOCOL),source_sha256_unchanged=True,checkpoint_sha256=checkpoint_sha))
  heartbeat.update(status='COMPLETE',stage_completed=len(results),job_id=None,control_step=None)
  log('STAGE_COMPLETE',stage=label,episodes=len(results),success=sum(r['success'] for r in results))
 gpu_lock.close()

def queue():
 # Explicit root-authorized primary queue only. Secondary endpoints never launch here.
 protocol,checkpoints=runtime_checks()
 lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 for stage in ['GT','A','B']:
  atomic_json(ROOT/'queue_status.json',dict(status='RUNNING',current_stage=stage,pid=os.getpid(),protocol_sha256=sha(PROTOCOL),updated_unix=time.time()))
  with (ROOT/f'console_{stage}.log').open('a') as out:
   process=subprocess.run([sys.executable,'-B','-u',str(Path(__file__)),'--run',stage],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
  if process.returncode:
   atomic_json(ROOT/'queue_status.json',dict(status='ERROR',current_stage=stage,returncode=process.returncode,protocol_sha256=sha(PROTOCOL),updated_unix=time.time()))
   raise RuntimeError('Stage failed; raw results retained: '+stage)
 atomic_json(ROOT/'queue_status.json',dict(status='COMPLETE',current_stage=None,protocol_sha256=sha(PROTOCOL),updated_unix=time.time(),primary_model_episodes=64,gt_episodes=20))
 log('PRIMARY_QUEUE_COMPLETE',gt=20,models=64);lock.close()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',choices=['GT','A','B']);parser.add_argument('--queue',action='store_true');parser.add_argument('--secondary-train-best',action='store_true');args=parser.parse_args()
 assert sum(bool(x) for x in [args.prepare,args.run,args.queue])==1
 if args.prepare:assert not args.secondary_train_best;prepare()
 elif args.queue:assert not args.secondary_train_best;queue()
 else:run(args.run,args.secondary_train_best)
if __name__=='__main__':
 try:main()
 except Exception:
  log('ERROR',traceback=traceback.format_exc());raise
