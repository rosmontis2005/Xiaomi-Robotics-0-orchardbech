"""Paired B/R continuations from fixed mined student states; train-state diagnostic only."""
import bootstrap
import json,os,time,traceback,fcntl
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from recovery_common import *
from collect_recovery import regenerate,healthy
from checkpoint_io import load_trainable_overlay
from recovery_dataset import RecoveryDataset
from mibot.data.datamodule.orchardbench_datamodule import OrchardBenchDataModule
from treesim.orchard_action import decode_targets,encode_window,EPS
OUT=ROOT/'evaluation_followup/version_01/takeovers';OUT.mkdir(parents=True,exist_ok=True)
def status(phase,**kw):write(OUT/'status.json',dict(status=phase,pid=os.getpid(),time=time.time(),**kw))
rows=[json.loads(l) for l in (ROOT/'recovery_manifest.jsonl').read_text().splitlines()]
selected=[]
for cat in ['R1','R2','R3']:selected.extend([r for r in rows if r['category']==cat][:2])
selected.append(next(r for r in rows if r['category']=='R3' and r['source_episode_id']!=selected[-1]['source_episode_id']))
# One different-source replacement after regeneration failure; original remains unavailable.
protocol=dict(selection=[r['candidate_id'] for r in selected],count_per_category=2,seeds=[42],role='Training-source recovery mechanism diagnostic; not held-out generalization',original_B_sha256=PROTOCOL['baseline']['sha256'],R_checkpoint_sha256=sha(ROOT/'training/checkpoints/step_4000_trainable.pt'),code_sha256=sha(Path(__file__)))
if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
else:write(OUT/'protocol.json',protocol)
def offline(policy,label):
 if (OUT/f'offline_{label}.json').exists():return
 dataset=RecoveryDataset(ROOT/'recovery_manifest.jsonl');dm=OrchardBenchDataModule(dict(processor_path=str(PROCESSOR)));result=[]
 for r in rows:
  t=dataset.episodes[r['annotation']]
  for frame in [0,15,29]:
   item=dataset.from_trajectory(t,frame);batch=dm.collate_fn([item]);gt=encode_window(t,frame);anchor=t['proprios']['ee_pos'][frame];rot=np.asarray(t['proprios']['ee_rotm'][frame]).reshape(3,3);gtxyz,gtrot,_=decode_targets(gt,anchor,rot)
   b={k:v.to('cuda:0') for k,v in batch.items()};b['action']=torch.zeros_like(b['action'])
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);z=policy.model.generate(b)[0].float().cpu().numpy()
   physical=z*(dataset.std+EPS)+dataset.mean;physical[:,7:]=0;xyz,rr,_=decode_targets(physical,anchor,rot)
   result.append(dict(candidate=r['candidate_id'],category=r['category'],frame=frame,anchor_phase=t['orchardbench']['phase_by_frame'][frame],position_mae_m=float(np.linalg.norm(xyz-gtxyz,axis=-1).mean()),rotation_mae_rad=float(Rotation.from_matrix(gtrot.transpose(0,2,1)@rr).magnitude().mean()),width_mae_m=float(np.abs(physical[:,6]-gt[:,6]).mean()),future_phases=dict(Counter(t['orchardbench']['phase_by_frame'][frame+1:frame+31]))))
 write(OUT/f'offline_{label}.json',result)
try:
 torch.set_num_threads(1);protection_check();lock=(ROOT/'.gpu_training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);status('MODEL_LOADING')
 from mibot.server.orchard_policy import OrchardPolicy
 policy=OrchardPolicy(str(BASE),str(PROCESSOR),str(STATS));policy.model.num_steps=5
 allresults=[]
 for label,checkpoint in [('B8000',ENDPOINT),('R4000',ROOT/'training/checkpoints/step_4000_trainable.pt')]:
  load=load_trainable_overlay(policy.model,checkpoint,base_checkpoint=BASE);write(OUT/f'load_{label}.json',load);offline(policy,label)
  for r in selected:
   folder=OUT/label/r['candidate_id'];folder.mkdir(parents=True,exist_ok=True)
   if (folder/'result.json').exists():allresults.append(json.loads((folder/'result.json').read_text()));continue
   c=json.loads((Path(r['annotation']).parent/'candidate.json').read_text());status('RUNNING',model=label,candidate=c['candidate_id'])
   try:
    if label=='R4000' and json.loads((OUT/'B8000'/r['candidate_id']/'result.json').read_text())['strict_success'] is None:raise RuntimeError('Paired B takeover unavailable; no unpaired outcome used')
    env,obs,tracker,prefix=regenerate(c)
   except Exception as e:
    result=dict(model=label,candidate=c['candidate_id'],category=c['category'],source_episode_id=c['source_episode_id'],strict_success=None,reason='takeover_regeneration_unavailable',error=str(e),traceback=traceback.format_exc());write(folder/'result.json',result);allresults.append(result);continue
   fruit=int(env._reset_stance['apple_index']);base=env.sim.body_q_np()[env.chassis].copy();initial=truth(env);steps=[];chunks=[];k=30;dwell=0;reason='budget' 
   try:
    for step in range(c['step']+1,901):
     if k==30:
      norm=policy.predict(obs,seed=42);xyz,rot,width=policy.adapter.targets;k=0;dwell=0;chunks.append(dict(step=step,position=xyz.tolist(),rotation=rot.tolist(),width=width.tolist()))
     command=policy.adapter.to_native(k,obs);obs,_,_,trunc,info=env.step(**command);dwell+=1;outcome=update_strict(env,tracker,step);adv,pe,re=advance(obs,xyz,rot,k,dwell);bad=healthy(env,base,fruit)
     steps.append(dict(step=step,chunk=len(chunks)-1,k=k,dwell=dwell,advance=adv,position_error=pe,rotation_error=re,target_width=float(width[k]),truth=truth(env),strict=outcome,ik_failed=info['ik_failed'],failure=bad))
     if bad:reason=bad;break
     if outcome['strict_success'] and any(e['apple_id']==fruit for e in outcome['strict_success_events']):reason='strict_success';break
     if adv:k+=1;dwell=0
     if trunc:break
    result=dict(model=label,candidate=c['candidate_id'],category=c['category'],source_episode_id=c['source_episode_id'],takeover=initial,prefix_regeneration=prefix,strict=tracker.summary(),strict_success=reason=='strict_success',reason=reason,continuation_steps=len(steps),replans=len(chunks),checkpoint_sha256=sha(checkpoint))
    write(folder/'steps.json',steps);write(folder/'chunks.json',chunks);write(folder/'result.json',result);allresults.append(result);print(json.dumps(dict(event='TAKEOVER_RESULT',model=label,candidate=c['candidate_id'],reason=reason)),flush=True)
   finally:env.close()
 write(OUT/'results.json',allresults);status('COMPLETE',results=len(allresults))
except BaseException as e:status('FAILED',error=str(e),traceback=traceback.format_exc());raise
