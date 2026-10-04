"""Finish paired reach after the independent precision panels; never redo training.

M0's validator checks ROOT/bootstrap.py. Validate while ROOT is this AB directory,
then relocate only the rollout output root. Preserve the initial validation log.
"""
import verify_precision_fix as v
from pathlib import Path
import json,gc,time
import numpy as np
import torch


def run():
 torch.set_num_threads(1)
 root,out,m0=v.ROOT,v.OUT,v.M0
 original=json.loads((out/'original_generation_comparison.json').read_text())
 overlay=json.loads((out/'overlay_generation_comparison.json').read_text())
 for group in (original['correct_source_runner_vs_policy'],overlay['correct_source_runner_vs_policy'],overlay['historical_m0_step2000']):
  assert len(group)==4 and all(r['normalized_exact_equal'] and r['physical_exact_equal'] for r in group)
 audits={k:json.loads((out/f'{k}_source_audit.json').read_text()) for k in ('runner_original','runner_overlay','policy_original','policy_overlay')}
 source=json.loads((root/'precision_source_backup/manifest.json').read_text());assert v.sha(source['source'])==source['after_sha256']
 protected=[m0/'selection.json',m0/'input_cache.pt',m0/'checkpoints/best_trainable.pt',m0/'checkpoints/final_trainable.pt',m0/'evaluate_reach.py',m0/'deployment_precision_probe.py',v.XR0/'mibot/server/orchard_policy.py',v.XR0/'mibot/models/runner/orchard_runner.py',v.XR0/'mibot/models/VLA/XR0.py',v.ORCHARD/'treesim/vla_env.py',v.ORCHARD/'treesim/orchard_action.py',v.STATS]
 protected_before={str(p):v.sha(p) for p in protected}
 payload,samples,selection=v.load_samples();mean,std=payload['mean'].numpy(),payload['std'].numpy()
 from mibot.server.orchard_policy import OrchardPolicy
 v.log('ROLLOUT_CONTINUATION_POLICY_LOADING')
 policy=OrchardPolicy(str(v.BASE),str(v.PROCESSOR),str(v.STATS),device='cpu')
 loader=v.module(m0/'checkpoint_io.py','precision_m0_overlay_reload')
 load_report=loader.load_trainable_overlay(policy.model,v.OVERLAY,base_checkpoint=v.BASE)
 assert v.dtypes(policy.model)=={'vlm':{'torch.bfloat16':713},'non_vlm':{'torch.float32':219}}
 policy.model.eval().to('cuda:0');policy.device='cuda:0'
 rows=v.generate_panel(policy.model,samples,mean,std,out/'policy_overlay_rollout_reload')
 reload_comparison=[v.compare(Path(r['path']),out/'policy_overlay'/Path(r['path']).name) for r in rows]
 assert all(r['normalized_exact_equal'] and r['physical_exact_equal'] for r in reload_comparison)
 v.save(out/'rollout_reload_exact.json',reload_comparison)
 rollout=v.module(m0/'evaluate_reach.py','precision_reused_m0_rollout')
 rollout.ROOT=root
 validation=rollout.validate_inputs(m0/'selection.json')
 rollout.ROOT=root/'precision_rollout';rollout.ROOT.mkdir(exist_ok=True)
 rollout.load_runtime();_,scenes=rollout.selected_scenes(m0/'selection.json')
 old_manifest=json.loads((m0/'evaluation_manifest_best.json').read_text());assert validation['reach_protocol']==old_manifest['reach_protocol']
 original_check=rollout.check_reset_scene
 def paired_check(obs,info,traj,directory,initial):
  report=original_check(obs,info,traj,directory,initial)
  previous=m0/'rollouts'/f"best_reach-conditioned_{traj['seed']}"/'initial.json'
  old=json.loads(previous.read_text())
  checks={k:(val==old['obs'][k] if k.startswith('rgb_') else bool(np.array_equal(np.asarray(val),np.asarray(old['obs'][k])))) for k,val in initial['obs'].items()}
  report['prior_bf16_m0_reset_pairing']=dict(path=str(previous),checks=checks,exact_match=all(checks.values()))
  report['state_and_identity_pass']=report['state_and_identity_pass'] and all(checks.values())
  return report
 rollout.check_reset_scene=paired_check
 v.save(rollout.ROOT/'manifest.json',dict(reused_rollout_source=str(m0/'evaluate_reach.py'),reused_rollout_source_sha256=v.sha(m0/'evaluate_reach.py'),formal_reach_protocol=validation['reach_protocol'],reference_bf16_manifest=str(m0/'evaluation_manifest_best.json'),reference_bf16_manifest_sha256=v.sha(m0/'evaluation_manifest_best.json'),precision_only_change=True,checkpoint=str(v.OVERLAY),checkpoint_sha256=v.sha(v.OVERLAY),load_report=load_report,source_fix=source,continuation_script_sha256=v.sha(Path(__file__))))
 summaries=[]
 for scene in scenes:
  v.log('PAIRED_ROLLOUT_START',seed=scene['seed'])
  result=rollout.episode(policy,'m0_fp32',scene,str(v.OVERLAY),True,validation['selection_sha256']);summaries.append(result)
  assert result['termination']!='error' and result['scene_check_pass'],result
  v.log('PAIRED_ROLLOUT_COMPLETE',seed=scene['seed'],grasp=result['ever_grasped'],detach=result['max_apple_detached_count'],success=result['success'],steps=result['control_steps'])
 del policy;gc.collect();torch.cuda.empty_cache()
 protected_after={str(p):v.sha(p) for p in protected};assert protected_before==protected_after
 pairs=[]
 for result in summaries:
  seed=result['seed'];priorpath=m0/'rollouts'/f'best_reach-conditioned_{seed}'/'summary.json';prior=json.loads(priorpath.read_text())
  fields=['ever_grasped','max_apple_detached_count','success','control_steps','ik_failed_steps','dwell_timeouts','replans']
  pairs.append(dict(seed=seed,split=result['split'],episode_id=result['episode_id'],previous_bf16={k:prior[k] for k in fields},fixed_fp32={k:result[k] for k in fields},previous_summary=str(priorpath),previous_summary_sha256=v.sha(priorpath)))
 result=dict(status='PASS',production_fix=source,cpu_validation=json.loads((out/'cpu_validation.json').read_text()),cpu_shape_validation=json.loads((out/'cpu_shape_validation.json').read_text()),restoration_audits={k:dict(non_vlm_tensors_exact=a['non_vlm_tensors_exact'],source_sha256=a['source_sha256'],source_tensors_that_would_lose_bf16_roundtrip=a['source_tensors_that_would_lose_bf16_roundtrip'],source_elements_that_would_lose_bf16_roundtrip=a['source_elements_that_would_lose_bf16_roundtrip'],parameter_dtypes=a['parameter_dtypes']) for k,a in audits.items()},original_runner_policy=original['correct_source_runner_vs_policy'],overlay_runner_policy=overlay['correct_source_runner_vs_policy'],historical_m0_original=original['historical_m0_step0'],historical_m0_overlay=overlay['historical_m0_step2000'],rollout_reload_exact=reload_comparison,rollout_summaries=summaries,paired_rollouts=pairs,protected_files_unchanged=True,protected_sha256=protected_before,entrypoint_note='Initial validation completed all precision panels, then its relocated output directory lacked bootstrap.py required by the reused validator. Continuation validates against the AB root before relocating rollout outputs, preserving all original/M0 source and results.',limits=['Only 4 fixed M0 scenes and one generation seed are evaluated; this is not a population success-rate estimate.','Historical M0 step0 used BF16-rounded source followed by FP32 cast; new original reference restores original checkpoint FP32 directly.','Model precision is fixed; reach, action contract, controller limits and grasp predicate are unchanged.'],gpu_released=True)
 v.save(root/'precision_fix.json',result);v.log('COMPLETE',status='PASS',grasp=sum(s['ever_grasped'] for s in summaries),detach=sum(s['max_apple_detached_count']>0 for s in summaries),success=sum(s['success'] for s in summaries),gpu_released=True)
if __name__=='__main__':
 try:run()
 except Exception:
  import traceback
  v.log('CONTINUATION_ERROR',traceback=traceback.format_exc());raise
