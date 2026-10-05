"""Fixed six train-source paired continuation diagnostics using an already loaded model."""
def evaluate(policy,original,output,checkpoint,heartbeat):
 import json
 from pathlib import Path
 from collections import Counter
 import torch,numpy as np
 from scipy.spatial.transform import Rotation
 from recovery_common import write,sha,truth,update_strict,advance
 from collect_recovery import regenerate,healthy
 from recovery_dataset import RecoveryDataset
 from mibot.data.datamodule.orchardbench_datamodule import OrchardBenchDataModule
 from treesim.orchard_action import encode_window,decode_targets,EPS
 original=Path(original);output=Path(output);output.mkdir(parents=True,exist_ok=True);baseline=original/'evaluation_followup/version_01/takeovers';protocol=json.loads((baseline/'protocol.json').read_text());rows=[json.loads(l) for l in (original/'recovery_manifest.jsonl').read_text().splitlines()]
 write(output/'protocol.json',dict(selection=protocol['selection'],B_results_reused=str(baseline/'B8000'),B_original_protocol_sha256=sha(baseline/'protocol.json'),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),role='Fixed original train-source continuation diagnosis, never generalization',code_sha256=sha(Path(__file__))))
 dataset=RecoveryDataset(original/'recovery_manifest.jsonl');dm=OrchardBenchDataModule(dict(processor_path=str(__import__('bootstrap').XR0.parent/'checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D')));offline=[]
 for row in rows:
  t=dataset.episodes[row['annotation']];item=dataset.from_trajectory(t,0);batch=dm.collate_fn([item]);gt=encode_window(t,0);anchor=t['proprios']['ee_pos'][0];rot=np.asarray(t['proprios']['ee_rotm'][0]).reshape(3,3);gtxyz,gtrot,_=decode_targets(gt,anchor,rot);batch={k:v.to('cuda:0') for k,v in batch.items()};batch['action']=torch.zeros_like(batch['action'])
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):torch.manual_seed(42);torch.cuda.manual_seed_all(42);z=policy.model.generate(batch)[0].float().cpu().numpy()
  a=z*(dataset.std+EPS)+dataset.mean;a[:,7:]=0;xyz,rr,_=decode_targets(a,anchor,rot)
  offline.append(dict(candidate=row['candidate_id'],category=row['category'],frame=0,position_mae_m=float(np.linalg.norm(xyz-gtxyz,axis=-1).mean()),first3_position_mae_m=float(np.linalg.norm(xyz[:3]-gtxyz[:3],axis=-1).mean()),first5_position_mae_m=float(np.linalg.norm(xyz[:5]-gtxyz[:5],axis=-1).mean()),rotation_mae_rad=float(Rotation.from_matrix(gtrot.transpose(0,2,1)@rr).magnitude().mean()),width_mae_m=float(np.abs(a[:,6]-gt[:,6]).mean())))
 write(output/'offline_R.json',offline);results=[]
 for cid in protocol['selection']:
  b=json.loads((baseline/'B8000'/cid/'result.json').read_text());row=next(r for r in rows if r['candidate_id']==cid);c=json.loads((Path(row['annotation']).parent/'candidate.json').read_text());folder=output/cid;folder.mkdir(parents=True,exist_ok=True);heartbeat('TAKEOVER',job=cid,completed=len(results))
  if b['strict_success'] is None:
   result=dict(candidate=cid,category=c['category'],strict_success=None,reason='Original paired B regeneration unavailable, retained');write(folder/'result.json',result);results.append(result);continue
  try:env,obs,tracker,prefix=regenerate(c)
  except Exception as e:
   result=dict(candidate=cid,category=c['category'],strict_success=None,reason='takeover_regeneration_unavailable',error=str(e));write(folder/'result.json',result);results.append(result);continue
  fruit=int(env._reset_stance['apple_index']);base=env.sim.body_q_np()[env.chassis].copy();steps=[];chunks=[];k=30;dwell=0;reason='budget';initial=truth(env)
  try:
   for step in range(c['step']+1,901):
    if k==30:
     policy.predict(obs,seed=42);xyz,rot,width=policy.adapter.targets;k=0;dwell=0;chunks.append(dict(step=step,position=xyz.tolist(),rotation=rot.tolist(),width=width.tolist()))
    obs,_,_,truncated,info=env.step(**policy.adapter.to_native(k,obs));dwell+=1;outcome=update_strict(env,tracker,step);adv,pe,re=advance(obs,xyz,rot,k,dwell);bad=healthy(env,base,fruit);steps.append(dict(step=step,chunk=len(chunks)-1,k=k,dwell=dwell,advance=adv,truth=truth(env),strict=outcome,position_error=pe,rotation_error=re,failure=bad))
    if step%100==0:heartbeat('TAKEOVER',job=cid,control_step=step,completed=len(results))
    if bad:reason=bad;break
    if outcome['strict_success'] and any(e['apple_id']==fruit for e in outcome['strict_success_events']):reason='strict_success';break
    if adv:k+=1;dwell=0
    if truncated:break
   held=[s for s in steps if s['truth']['held']];result=dict(candidate=cid,category=c['category'],source_episode_id=c['source_episode_id'],takeover=initial,prefix_regeneration=prefix,strict_success=reason=='strict_success',strict=tracker.summary(),reason=reason,continuation_steps=len(steps),replans=len(chunks),minimum_bucket_distance_while_held=min([s['truth']['fruit_bucket_distance'] for s in held],default=None),B_strict=b['strict_success'],B_result=str(baseline/'B8000'/cid/'result.json'),checkpoint_sha256=sha(checkpoint))
   write(folder/'steps.json',steps);write(folder/'chunks.json',chunks);write(folder/'result.json',result);results.append(result);print(json.dumps(dict(event='FIXED_TAKEOVER',candidate=cid,strict=result['strict_success'],reason=reason)),flush=True)
  finally:env.close()
 write(output/'results.json',results);return results
