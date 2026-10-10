from common import *
from treesim.orchard_command import CONTRACT

def run():
 if (ROOT/'frozen.json').exists():
  for p,h in json.loads((ROOT/'frozen.json').read_text())['artifacts'].items():assert sha(p)==h,p
  return
 assert (ROOT/'visibility_summary.json').exists()
 cfg=dict(contract=CONTRACT,max_optimizer_updates=60000,batch_size=1,gradient_accumulation=1,optimizer='AdamW',lr=1e-5,betas=[.9,.95],weight_decay=.1,eps=1e-8,foreach=False,gradient_clip=1.,seed=42,lr_schedule='constant',pretrained=str(BASE),initialization='weights-only; direct source to FP32 non-VLM master',frozen_vlm=True,vlm_dtype='bfloat16',non_vlm_dtype='float32',autocast='bfloat16',action_shape=[30,32],active_dims=list(range(7)),training_repeat=1,enable_freq=False,async_train=False,inference_euler_steps=5,execution_horizon=5,controller_budget=2100,phase_ratios=dict(reset_early=.20,grasp=.20,pull=.10,transport=.25,drop=.20,transition=.05),reset_fraction=.05,checkpoints=[0,2000,5000,10000,20000,40000,60000],closed_loop_checkpoints=[0,10000,20000,40000,60000],fixed_panel_per_split=256,fixed_loss_interval=2000,resume_interval=500,closed_loop_inference_seed=42,lateral_offsets=[0,-.08,.08],strict_observer='original held15/detach/actual release/non-held/bucket60; reasonable release independent diagnostic',no_second_training=True)
 write(ROOT/'config.json',cfg)
 artifacts=[ROOT/n for n in ['config.json','train_manifest.jsonl','validation_manifest.jsonl','action_stats.json','schedule.jsonl','panels.jsonl','scenes.json','schedule_revision.json']]
 artifacts+=list(ROOT.glob('*.py'))
 artifacts += [XR/'mibot/models/VLA/XR0.py',XR/'mibot/utils/orchard_checkpoint.py',XR/'mibot/data/collate/custom_collate.py',ORCHARD/'treesim/orchard_command.py',ORCHARD/'treesim/orchard_action.py',ORCHARD/'treesim/student_native_expert.py',XR/'test_grasp_1004/evaluation/strict_metrics.py']
 write(ROOT/'frozen.json',dict(artifacts={str(p):sha(p) for p in artifacts},original_accepted_manifest_sha256=sha(DATA/'accepted_manifest.jsonl'),pretrained={str(p):sha(p) for p in BASE.glob('*.safetensors')},created_before_model_results=True))
if __name__=='__main__':run()
