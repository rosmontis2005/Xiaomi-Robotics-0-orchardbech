"""Verify Orchard FP32 source restoration, independent runner/policy generation and paired reach.

Reads the immutable M0 panel. All new outputs/caches live in this AB directory.
--cpu checks the actual loader on values that a BF16 roundtrip would destroy.
--run uses one model on GPU at a time; runs no optimizer/training updates.
"""
import bootstrap
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import importlib.util
import json
import gc
import time
import traceback
import ast
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT=bootstrap.ROOT;XR0=bootstrap.XR0;ORCHARD=bootstrap.ORCHARD
M0=XR0/'test_small_set_fit_1003'
BASE=XR0/'outputs/orchard_v1_2650/project_orchardbench/orchard_v1_2650_frozen_vlm_seed42/epoch=0-step=10000.ckpt'
OVERLAY=M0/'checkpoints/best_trainable.pt'
STATS=ORCHARD/'data/orchard_v1_2650/filtered/action_stats.json'
PROCESSOR=XR0.parent/'checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D'
OUT=ROOT/'precision_validation'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()

def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def log(event,**data):
 row=dict(unix=time.time(),event=event,**data)
 with (ROOT/'precision_fix.log').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)

def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def cpu_check():
 from mibot.utils.orchard_checkpoint import load_weights
 assert not torch.cuda.is_initialized()
 class TinyOrchard(torch.nn.Module):
  def __init__(self):
   super().__init__();self.vlm=torch.nn.Linear(2,2,bias=False).bfloat16();self.action_head=torch.nn.Linear(2,2,bias=False).bfloat16()
 source={'vlm.weight':torch.tensor([[1.125,.125],[.5,-.25]],dtype=torch.bfloat16),
         'action_head.weight':torch.tensor([[1.00017,.0100137],[-.99137,.234567]],dtype=torch.float32)}
 fixture=OUT/'cpu_source_checkpoint.pt';fixture.parent.mkdir(parents=True,exist_ok=True)
 with fixture.open('xb') as f:torch.save({'state_dict':{'model.'+k:v for k,v in source.items()}},f)
 model=TinyOrchard();report=load_weights(model,fixture)
 assert torch.equal(model.action_head.weight,source['action_head.weight'])
 assert model.action_head.weight.dtype==torch.float32 and model.vlm.weight.dtype==torch.bfloat16
 assert not torch.equal(source['action_head.weight'],source['action_head.weight'].bfloat16().float())
 missing=OUT/'cpu_missing_key.pt'
 with missing.open('xb') as f:torch.save({'state_dict':{'model.vlm.weight':source['vlm.weight']}},f)
 try:load_weights(TinyOrchard(),missing)
 except RuntimeError:strict_missing_rejected=True
 else:raise AssertionError('Strict missing-key rejection was lost')
 assert not torch.cuda.is_initialized()
 result=dict(status='PASS',cuda_initialized=False,source_fp32_exact=True,vlm_bf16_unchanged=True,
             distinguishes_lossy_roundtrip=True,strict_missing_rejected=strict_missing_rejected,load_report=report)
 save(OUT/'cpu_validation.json',result);log('CPU_VALIDATION',**result);return result

def dtypes(model):
 return {part:dict(Counter(str(p.dtype) for n,p in model.named_parameters() if n.startswith('vlm.')==isvlm))
         for part,isvlm in [('vlm',True),('non_vlm',False)]}

def audit_source(model,source_path,overlay=False):
 payload=torch.load(source_path,map_location='cpu',weights_only=True,mmap=True)
 state=payload['trainable_state_dict'] if overlay else {k.removeprefix('model.'):v for k,v in payload.get('module',payload.get('state_dict',payload)).items()}
 rows=[];lossy_tensors=0;lossy_elements=0
 for name,p in model.named_parameters():
  if name.startswith('vlm.'):continue
  expected=state[name];actual=p.detach().cpu()
  assert expected.dtype==torch.float32,(name,expected.dtype)
  assert actual.dtype==torch.float32 and torch.equal(actual,expected),name
  count=int((expected!=expected.bfloat16().float()).sum());lossy_tensors+=bool(count);lossy_elements+=count
  rows.append(dict(name=name,shape=list(p.shape),source_dtype=str(expected.dtype),destination_dtype=str(p.dtype),exact_equal=True,max_abs_difference=0.0))
 assert len(rows)==219
 kinds=dtypes(model);assert kinds=={'vlm':{'torch.bfloat16':713},'non_vlm':{'torch.float32':219}},kinds
 result=dict(source=str(source_path),source_sha256=sha(source_path),non_vlm_tensors_exact=len(rows),non_vlm_elements=sum(p.numel() for n,p in model.named_parameters() if not n.startswith('vlm.')),
             source_tensors_that_would_lose_bf16_roundtrip=lossy_tensors,source_elements_that_would_lose_bf16_roundtrip=lossy_elements,parameter_dtypes=kinds,parameters=rows)
 del payload,state;gc.collect();return result

def load_samples():
 payload=torch.load(M0/'input_cache.pt',map_location='cpu',weights_only=True,mmap=True)
 selection=json.loads((M0/'selection.json').read_text());assert payload['provenance']['selection_sha256']==sha(M0/'selection.json')
 samples=[]
 for scene in selection['closed_loop_scenes']:
  found=[s for s in payload['samples'] if s['meta']['episode_id']==scene['episode_id'] and s['meta']['split']==scene['split'] and s['meta']['frame']==0]
  assert len(found)==1;s=found[0];assert s['meta']['seed']==scene['seed'];samples.append(s)
 return payload,samples,selection

def generate_panel(model,samples,mean,std,path):
 from treesim.orchard_action import EPS
 path.mkdir(parents=True,exist_ok=False);rows=[]
 model.eval();assert model.num_steps==5 and not model.async_train and model.training_repeat==1
 with torch.inference_mode(),torch.random.fork_rng(devices=[0]):
  for sample in samples:
   batch={k:(torch.zeros_like(v) if k=='action' else v).to('cuda:0') for k,v in sample['batch'].items()}
   assert 'prefix_length' not in batch and not torch.count_nonzero(batch['action']).item()
   torch.manual_seed(42);torch.cuda.manual_seed_all(42)
   with torch.autocast('cuda',dtype=torch.bfloat16):z=model.generate(batch)[0].float().cpu().numpy()
   z[:,7:]=0;physical=z*(std+EPS)+mean;physical[:,7:]=0
   assert np.isfinite(z).all();meta=sample['meta'];file=path/f"{meta['split']}_{meta['episode_id']}_frame0000.npz"
   with file.open('xb') as f:np.savez_compressed(f,normalized=z,physical=physical,prediction_seed=np.array(42))
   rows.append(dict(seed=meta['seed'],episode_id=meta['episode_id'],split=meta['split'],path=str(file),sha256=sha(file)))
   del batch
 return rows

def compare(actual_path,reference_path,historical=False):
 with np.load(actual_path,allow_pickle=False) as a:az=a['normalized'];aa=a['physical']
 with np.load(reference_path,allow_pickle=False) as b:
  if historical:
   at=int(np.flatnonzero(b['prediction_seeds']==42)[0]);bz=b['normalized_prediction'][at];ba=b['predicted_action_physical'][at]
  else:bz=b['normalized'];ba=b['physical']
 delta=np.abs(az[:,:7].astype(float)-bz[:,:7].astype(float));pd=np.linalg.norm(aa[:,:3]-ba[:,:3],axis=-1);rd=(Rotation.from_rotvec(ba[:,3:6]).inv()*Rotation.from_rotvec(aa[:,3:6])).magnitude();wd=np.abs(aa[:,6]-ba[:,6])
 return dict(actual=str(actual_path),reference=str(reference_path),reference_sha256=sha(reference_path),normalized_exact_equal=bool(np.array_equal(az,bz)),normalized_mean_abs=float(delta.mean()),normalized_max_abs=float(delta.max()),physical_exact_equal=bool(np.array_equal(aa,ba)),position_mean_m=float(pd.mean()),position_max_m=float(pd.max()),rotation_max_rad=float(rd.max()),width_max_m=float(wd.max()))

def run():
 if (ROOT/'precision_fix.json').exists():raise FileExistsError('Precision result exists; refuse overwriting evidence')
 torch.set_num_threads(1);log('GPU_START',gpu=torch.cuda.get_device_name(0))
 payload,samples,selection=load_samples();mean,std=payload['mean'].numpy(),payload['std'].numpy()
 protected=[M0/'selection.json',M0/'input_cache.pt',M0/'checkpoints/best_trainable.pt',M0/'checkpoints/final_trainable.pt',M0/'evaluate_reach.py',M0/'deployment_precision_probe.py',XR0/'mibot/server/orchard_policy.py',XR0/'mibot/models/runner/orchard_runner.py',XR0/'mibot/models/VLA/XR0.py',ORCHARD/'treesim/vla_env.py',ORCHARD/'treesim/orchard_action.py',STATS]
 protected_before={str(p):sha(p) for p in protected}
 from mibot.models.runner.orchard_runner import OrchardRunner
 overlay_io=module(M0/'checkpoint_io.py','precision_m0_overlay_io')
 params=dict(pretrained=str(BASE),freeze_vlm=True,diagnostic_path=None,
    model=dict(type='XR0',vlm_config_path=str(PROCESSOR/'config.json'),training_repeat=1,enable_freq=False,async_train=False))
 log('RUNNER_LOADING');runner=OrchardRunner(params);runner.configure_model();model=runner.model
 audits={'runner_original':audit_source(model,BASE)};save(OUT/'runner_original_source_audit.json',audits['runner_original'])
 assert all(not p.requires_grad for p in model.vlm.parameters())
 model.to('cuda:0');generate_panel(model,samples,mean,std,OUT/'runner_original');log('RUNNER_ORIGINAL_GENERATED')
 overlay_io.load_trainable_overlay(model,OVERLAY,base_checkpoint=BASE)
 audits['runner_overlay']=audit_source(model,OVERLAY,True);save(OUT/'runner_overlay_source_audit.json',audits['runner_overlay'])
 generate_panel(model,samples,mean,std,OUT/'runner_overlay');log('RUNNER_OVERLAY_GENERATED')
 del model,runner;gc.collect();torch.cuda.empty_cache()
 from mibot.server.orchard_policy import OrchardPolicy
 log('POLICY_LOADING');policy=OrchardPolicy(str(BASE),str(PROCESSOR),str(STATS),device='cpu')
 audits['policy_original']=audit_source(policy.model,BASE);save(OUT/'policy_original_source_audit.json',audits['policy_original'])
 assert np.array_equal(policy.mean,mean) and np.array_equal(policy.std,std)
 policy.model.to('cuda:0');policy.device='cuda:0'
 panels={'original':generate_panel(policy.model,samples,mean,std,OUT/'policy_original')}
 original_pairs=[];historical_original=[]
 for row in panels['original']:
  file=Path(row['path']);original_pairs.append(compare(file,OUT/'runner_original'/file.name));historical_original.append(compare(file,M0/'evaluations/step_0000'/file.name,True))
 assert all(v['normalized_exact_equal'] and v['physical_exact_equal'] for v in original_pairs),'Correct-source runner/policy differ'
 save(OUT/'original_generation_comparison.json',dict(correct_source_runner_vs_policy=original_pairs,historical_m0_step0=historical_original))
 log('ORIGINAL_GENERATION_PARITY',exact_windows=4,historical_normalized_max_abs=max(v['normalized_max_abs'] for v in historical_original))
 overlay_report=overlay_io.load_trainable_overlay(policy.model,OVERLAY,base_checkpoint=BASE)
 audits['policy_overlay']=audit_source(policy.model,OVERLAY,True);save(OUT/'policy_overlay_source_audit.json',audits['policy_overlay'])
 panels['overlay']=generate_panel(policy.model,samples,mean,std,OUT/'policy_overlay');overlay_pairs=[];historical_overlay=[]
 for row in panels['overlay']:
  file=Path(row['path']);overlay_pairs.append(compare(file,OUT/'runner_overlay'/file.name));historical_overlay.append(compare(file,M0/'evaluations/step_2000'/file.name,True))
 assert all(v['normalized_exact_equal'] and v['physical_exact_equal'] for v in overlay_pairs),'Overlay runner/policy differ'
 assert all(v['normalized_exact_equal'] and v['physical_exact_equal'] for v in historical_overlay),'FP32 overlay must match original M0 master panel'
 save(OUT/'overlay_generation_comparison.json',dict(correct_source_runner_vs_policy=overlay_pairs,historical_m0_step2000=historical_overlay))
 log('OVERLAY_GENERATION_PARITY',exact_runner_windows=4,exact_historical_m0_windows=4)
 # Reuse the frozen M0 rollout implementation itself; relocate only output root.
 rollout=module(M0/'evaluate_reach.py','precision_frozen_m0_rollout');rollout.ROOT=ROOT/'precision_rollout';rollout.ROOT.mkdir(exist_ok=False)
 validation=rollout.validate_inputs(M0/'selection.json');rollout.load_runtime();_,scenes=rollout.selected_scenes(M0/'selection.json')
 old_manifest=json.loads((M0/'evaluation_manifest_best.json').read_text());assert validation['reach_protocol']==old_manifest['reach_protocol']
 original_check=rollout.check_reset_scene
 def paired_check(obs,info,traj,directory,initial):
  report=original_check(obs,info,traj,directory,initial)
  old=json.loads((M0/'rollouts'/f"best_reach-conditioned_{traj['seed']}"/'initial.json').read_text())
  checks={k:(v==old['obs'][k] if k.startswith('rgb_') else bool(np.array_equal(np.asarray(v),np.asarray(old['obs'][k])))) for k,v in initial['obs'].items()}
  report['prior_bf16_m0_reset_pairing']=dict(path=str(M0/'rollouts'/f"best_reach-conditioned_{traj['seed']}"/'initial.json'),checks=checks,exact_match=all(checks.values()))
  report['state_and_identity_pass']=report['state_and_identity_pass'] and all(checks.values())
  return report
 rollout.check_reset_scene=paired_check
 save(rollout.ROOT/'manifest.json',dict(reused_rollout_source=str(M0/'evaluate_reach.py'),reused_rollout_source_sha256=sha(M0/'evaluate_reach.py'),formal_reach_protocol=validation['reach_protocol'],reference_bf16_manifest=str(M0/'evaluation_manifest_best.json'),reference_bf16_manifest_sha256=sha(M0/'evaluation_manifest_best.json'),precision_only_change=True,checkpoint=str(OVERLAY),checkpoint_sha256=sha(OVERLAY),load_report=overlay_report,source_fix=json.loads((ROOT/'precision_source_backup/manifest.json').read_text())))
 summaries=[]
 for scene in scenes:
  log('PAIRED_ROLLOUT_START',seed=scene['seed'])
  result=rollout.episode(policy,'m0_fp32',scene,str(OVERLAY),True,validation['selection_sha256']);summaries.append(result)
  assert result['termination']!='error' and result['scene_check_pass'],result
  log('PAIRED_ROLLOUT_COMPLETE',seed=scene['seed'],grasp=result['ever_grasped'],detach=result['max_apple_detached_count'],success=result['success'],steps=result['control_steps'])
 del policy;gc.collect();torch.cuda.empty_cache()
 protected_after={str(p):sha(p) for p in protected};assert protected_before==protected_after
 result=dict(status='PASS',production_fix=json.loads((ROOT/'precision_source_backup/manifest.json').read_text()),cpu_validation=json.loads((OUT/'cpu_validation.json').read_text()),cpu_shape_validation=json.loads((OUT/'cpu_shape_validation.json').read_text()),restoration_audits={k:dict(non_vlm_tensors_exact=v['non_vlm_tensors_exact'],source_sha256=v['source_sha256'],source_tensors_that_would_lose_bf16_roundtrip=v['source_tensors_that_would_lose_bf16_roundtrip'],source_elements_that_would_lose_bf16_roundtrip=v['source_elements_that_would_lose_bf16_roundtrip'],parameter_dtypes=v['parameter_dtypes']) for k,v in audits.items()},original_runner_policy=original_pairs,overlay_runner_policy=overlay_pairs,historical_m0_original=historical_original,historical_m0_overlay=historical_overlay,rollout_summaries=summaries,protected_files_unchanged=True,protected_sha256=protected_before,limits=['Only 4 fixed M0 scenes and one generation seed are evaluated; this is not a population success-rate estimate.','Historical M0 step0 used BF16-rounded source followed by FP32 cast; new original reference restores original checkpoint FP32 directly.','Model precision is fixed; reach, action contract, controller limits and grasp predicate are unchanged.'],gpu_released=True)
 save(ROOT/'precision_fix.json',result);log('COMPLETE',status='PASS',grasp=sum(s['ever_grasped'] for s in summaries),detach=sum(s['max_apple_detached_count']>0 for s in summaries),success=sum(s['success'] for s in summaries),gpu_released=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--cpu',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
 if a.cpu:cpu_check()
 if a.run:run()
 if not a.cpu and not a.run:p.error('Choose --cpu or --run')
if __name__=='__main__':
 try:main()
 except Exception:
  log('ERROR',traceback=traceback.format_exc());raise
