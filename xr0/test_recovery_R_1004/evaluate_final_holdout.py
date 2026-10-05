"""One final frozen R endpoint versus B on existing heldout12. No tuning afterwards."""
import bootstrap
import argparse,json,os,sys,time,traceback,fcntl
from pathlib import Path
from collections import Counter
import torch
from recovery_common import *
from checkpoint_io import load_trainable_overlay
AB=XR0/'test_128_episode_AB_1003';bootstrap.AB=AB;D0=XR0/'test_grasp_1004/evaluation';sys.path.append(str(D0))
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--version',required=True);a=p.parse_args();OUT=ROOT/'evaluation_followup/final_holdout12';OUT.mkdir(parents=True,exist_ok=True)
def status(phase,**kw):write(OUT/'status.json',dict(status=phase,pid=os.getpid(),time=time.time(),**kw))
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
try:
 torch.set_num_threads(1);protection_check();lock=(ROOT/'.gpu_training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 selection=json.loads((AB/'selection.json').read_text());scenes=selection['heldout_scenes'];assert len(scenes)==12 and all(s['split']=='val' for s in scenes)
 d0=json.loads((D0/'protocol.json').read_text())
 for path,h in d0['source_sha256'].items():assert sha(path)==h,path
 forbidden={s['episode_id'] for s in scenes};assert not forbidden & set(selection['train_episode_ids'])
 checkpoint=Path(a.checkpoint);payload=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True);assert payload['step']==4000;del payload
 protocol=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),version=a.version,base_sha256=PROTOCOL['baseline']['base_sha256'] if 'base_sha256' in PROTOCOL['baseline'] else sha(BASE),B_checkpoint=str(ENDPOINT),B_sha256=sha(ENDPOINT),selection_sha256=sha(AB/'selection.json'),stats_sha256=sha(STATS),scenes=scenes,inference_seed=42,reach=d0['reach'],strict=d0['strict'],code_sha256=sha(Path(__file__)),endpoint_frozen_before_test=True,selection_rules=json.loads((ROOT/'evaluation_followup/selection_rules.json').read_text()),limits='Curated historical replay-PASS val cohort, not fresh random-seed population. Scene2010600 already used as an auxiliary diagnosis; report remaining11 separately. No heldout or fresh24 training sources.',further_training_after_test=False)
 if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
 else:write(OUT/'protocol.json',protocol)
 status('MODEL_LOADING');from mibot.server.orchard_policy import OrchardPolicy
 policy=OrchardPolicy(str(BASE),str(PROCESSOR),str(STATS));assert policy.model.num_steps==5
 loop=module(D0/'reach_loop.py','final_R_loop');loop.ROOT=OUT;loop.load_runtime();helpers=module(AB/'closed_loop_1004/evaluate_ab.py','final_helpers');helpers.ROOT=OUT;analyze=module(XR0/'diagnosis_1003/analyze_rollouts.py','final_analyze');passive=module(AB/'closed_loop_1004/passive_metrics.py','final_passive')
 active={};base_line=loop.line;base_snapshot=loop.snapshot_grasp;results=[]
 def line(stream,data):
  data.update(inference_seed=42,job_id=active['job_id']);base_line(stream,data)
  if 'control_step' in data and data['control_step']%100==0:status('CLOSED_LOOP',job=active['job_id'],control_step=data['control_step'],completed=len(results))
 def snapshot(env):
  snap=base_snapshot(env);idx=active['apple'];body=int(env.tm.apple_bodies[idx]);snap['planned_apple']=passive.planned_apple_observation(snap,idx,body,env.sim.body_q_np()[body,:3]);return snap
 loop.line=line;loop.snapshot_grasp=snapshot
 for label,weights in [('B8000',ENDPOINT),('R_selected',checkpoint)]:
  loaded=load_trainable_overlay(policy.model,weights,base_checkpoint=BASE);assert loaded['step']==(8000 if label=='B8000' else 4000);write(OUT/f'load_{label}.json',loaded)
  dtype=Counter(str(v.dtype) for n,v in policy.model.named_parameters() if not n.startswith('vlm.'));assert dtype=={'torch.float32':219}
  for scene in scenes:
   provider=f'{label}_targets30_rng42';job=f'{provider}_reach-conditioned_{scene["seed"]}';folder=OUT/'rollouts'/job
   if (folder/'completed.json').exists():results.append(json.loads((folder/'summary.json').read_text()));continue
   assert not folder.exists(),('Partial retained',folder);assert sha(scene['annotation'])==scene['annotation_sha256'];traj=json.loads(Path(scene['annotation']).read_text());active.clear();active.update(job_id=job,apple=int(traj['orchardbench']['fixed_base_expert']['selected_apple_debug_index']));status('CLOSED_LOOP',job=job,completed=len(results),control_step=0)
   bound=helpers.BoundSeedPolicy(policy,42);result=loop.episode(bound,provider,scene,str(weights),True,sha(AB/'selection.json'),processed_targets=30,is_gt=False);assert result['termination']!='error';steps=lines(folder/'steps.jsonl');chunks=lines(folder/'chunks.jsonl');assert len(bound.calls)==result['replans']==len(chunks)
   metrics=analyze.analyze_episode(result,steps,chunks,folder);metrics.update(helpers.held_runs(steps));gt=AB/'closed_loop_1004/rollouts'/f'gt_reach-conditioned_{scene["seed"]}'/'chunks.jsonl';metrics.update(passive.diagnostics(steps,chunks,lines(gt)));metrics.update(strict_success=result['strict_success'],strict_placement=result['strict_placement'])
   assert result['scene_check_pass'] and metrics['observer_consistency']['mismatch_fields_total']==0 and metrics['summary_consistency']['mismatch_count']==0
   if label=='R_selected':
    reference=OUT/'rollouts'/f'B8000_targets30_rng42_reach-conditioned_{scene["seed"]}';assert json.loads((folder/'initial.json').read_text())['obs']==json.loads((reference/'initial.json').read_text())['obs']
   write(folder/'metrics.json',metrics);write(folder/'seed_delivery.json',dict(calls=bound.calls));write(folder/'completed.json',dict(status='COMPLETE',checkpoint_sha256=sha(weights),paired_reset_exact=label=='R_selected',files_sha256={n:sha(folder/n) for n in ['summary.json','metrics.json','steps.jsonl','chunks.jsonl','initial.json']}));results.append(result);print(json.dumps(dict(event='HOLDOUT_RESULT',job=job,strict=result['strict_success'],held15=metrics['same_fruit_held_at_least_15_steps'])),flush=True)
 write(OUT/'results.json',results);status('COMPLETE',completed=len(results));protection_check()
except BaseException as e:status('FAILED',error=str(e),traceback=traceback.format_exc());raise
