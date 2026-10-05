"""Budget-matched H conditions, using original R03 optimizer and flow objective."""
import argparse,os
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=['H0','Hrepeat','Hhistory','Hanchor'],required=True);parser.add_argument('--updates',type=int,choices=[4000,8000],required=True);parser.add_argument('--resume');parser.add_argument('--run',action='store_true');args=parser.parse_args()
os.environ['H_RUN_DIR']=str(Path(__file__).resolve().parent/args.condition)
import bootstrap
import argparse,fcntl,json,os,random,time,traceback,gc
from contextlib import contextmanager
from collections import Counter
import numpy as np
import torch
from recovery_common import ROOT,XR0,BASE,ENDPOINT,STATS,PROCESSOR,PROTOCOL,write,sha,protection_check
from history_inputs import HistoryInputs
ScheduledInputs=lambda rows: HistoryInputs(rows,args.condition)
from checkpoint_io import file_sha256,load_trainable_overlay,save_trainable_checkpoint,frozen_parameter_sha256
CONFIG=dict(updates=args.updates,checkpoint_steps=[args.updates],optimizer=dict(lr=1e-5,betas=[.9,.95],weight_decay=.1,eps=1e-8,foreach=False),batch=1,gradient_accumulation=1,gradient_clip_norm=1.,lr_schedule='constant',training_repeat=1,async_train=False,enable_freq=False,freeze_vlm=True,vlm_dtype='bfloat16',non_vlm_master_dtype='float32',autocast='bfloat16',euler_steps=5,loss='original full30 active7 normalized flow, coefficient0.5, no event/contact weighting',seed=42)

@contextmanager
def preserve_rng():
 py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
 try:yield
 finally:
  random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu)
  if cuda is not None:torch.cuda.set_rng_state_all(cuda)

def check_prepared():
 rows=[json.loads(l) for l in (ROOT/'schedule_R.jsonl').read_text().splitlines()]
 assert [r['step'] for r in rows]==list(range(1,args.updates+1))
 assert Counter(r['source'] for r in rows)=={k:v*(args.updates//4000) for k,v in dict(original_B=2800,R1=500,R2=200,R3=200,R_continuation_teacher=300).items()}
 return rows,dict(status='PASS',schedule_sha256=sha(ROOT/'schedule_R.jsonl'))

def construct_model():
 from mibot.models import MIMODEL
 from mibot.utils.orchard_checkpoint import load_weights
 model=MIMODEL.build(dict(type='XR0',vlm_config_path=str(PROCESSOR/'config.json'),training_repeat=1,enable_freq=False,async_train=False))
 for n,p in model.named_parameters():
  if not n.startswith('vlm.'):p.data=p.data.float()
 base_report=load_weights(model,str(BASE));model.vlm.requires_grad_(False);model.vlm.eval()
 payload=torch.load(BASE,map_location='cpu',weights_only=True,mmap=True);raw=payload.get('module',payload.get('state_dict',payload));source={k.removeprefix('model.'):v for k,v in raw.items()}
 for n,p in model.named_parameters():
  if not n.startswith('vlm.'):assert p.dtype==torch.float32 and source[n].dtype==torch.float32 and torch.equal(p.detach().cpu(),source[n]),n
 del source,raw,payload;gc.collect()
 overlay=load_trainable_overlay(model,ENDPOINT,base_checkpoint=BASE);assert overlay['step']==8000 and overlay['checkpoint_sha256']==PROTOCOL['baseline']['sha256']
 payload=torch.load(ENDPOINT,map_location='cpu',weights_only=True,mmap=True)
 for n,p in model.named_parameters():
  if not n.startswith('vlm.'):assert p.dtype==torch.float32 and p.requires_grad and torch.equal(p.detach().cpu(),payload['trainable_state_dict'][n]),n
 assert all(p.dtype==torch.bfloat16 and not p.requires_grad for p in model.vlm.parameters())
 assert model.num_steps==5 and model.flow_sampling=='beta' and model.training_repeat==1 and not model.async_train and model.freq_coefficient==0
 return model,dict(base=base_report,overlay=overlay,all_non_vlm_parameters_exactly_B8000=True,full_base_loaded_before_overlay=True)

def run():
 out=ROOT/'training';out.mkdir(exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
 if (out/'run_manifest.json').exists():raise FileExistsError('R training already initialized; refuse duplicate launch')
 lock=(ROOT/'.gpu_training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 oldlock=(XR0/'test_128_episode_AB_1003/.gpu_training.lock').open('r');fcntl.flock(oldlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 step=0;started=time.monotonic()
 def status(phase,**extra):write(out/'status.json',dict(status=phase,pid=os.getpid(),session=os.getsid(0),completed_updates=step,target_updates=args.updates,time=time.time(),elapsed_seconds=time.monotonic()-started,**extra))
 status('INITIALIZING')
 try:
  torch.set_num_threads(1);rows,prepared=check_prepared();random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  model,load_report=construct_model();frozen=frozen_parameter_sha256(model);initial=model.action_output_layer.layers[2].weight.detach().cpu().clone();model.to('cuda:0');model.train();model.vlm.eval()
  from mibot.models.runner.base_runner import BaseRunner
  options=dict(CONFIG['optimizer']);options['betas']=tuple(options['betas']);optimizer=BaseRunner.build_optimizer(dict(type='torch.optim.AdamW',params=options),((f'model.{n}',p) for n,p in model.named_parameters()))
  assert len(optimizer.state)==0 and all(p.requires_grad for group in optimizer.param_groups for p in group['params'])
  scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer,factor=1.,total_iters=1);trainable=[p for p in model.parameters() if p.requires_grad]
  resume_report=None
  if args.resume:
   resume=torch.load(args.resume,map_location='cpu',weights_only=False,mmap=True);step=resume['step'];assert step==4000 and args.updates==8000
   with torch.no_grad():
    for n,v in resume['trainable_state_dict'].items():dict(model.named_parameters())[n].copy_(v)
   optimizer.load_state_dict(resume['optimizer']);scheduler.load_state_dict(resume['scheduler'])
   torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng']);random.setstate(resume['python_rng']);np.random.set_state(resume['numpy_rng'])
   assert len(optimizer.state)==219 and all(float(s['step'])==step for s in optimizer.state.values())
   resume_report=dict(path=args.resume,sha256=sha(args.resume),step=step,optimizer_state_entries=len(optimizer.state),all_optimizer_steps=step,RNG_restored=True)
   del resume
  provenance=dict(baseline=PROTOCOL['baseline'],schedule_sha256=sha(ROOT/'schedule_R.jsonl'),manifest_sha256=sha(ROOT/'recovery_manifest.jsonl'),stats_sha256=sha(STATS),config=CONFIG,condition=args.condition,resume=resume_report,source_sha256={str(bootstrap.EXPERIMENT_ROOT/p):sha(bootstrap.EXPERIMENT_ROOT/p) for p in ['train_H.py','history_inputs.py','bootstrap.py']})
  write(out/'run_manifest.json',dict(configuration=CONFIG,provenance=provenance,initial_load=load_report,fresh_optimizer_state_entries=0 if resume_report is None else None,resume=resume_report,vlm_frozen=True,non_vlm_fp32=True))
  with preserve_rng():inputs=ScheduledInputs(rows)
  # One collated batch and one inference through actual dynamic token path.
  with preserve_rng(),torch.inference_mode():
   cpu=inputs.get(0);assert cpu['state'].shape==(1,1 if args.condition=='H0' else 4,32) and cpu['state'].dtype==torch.float32
   check={k:v.to('cuda:0') for k,v in cpu.items()};check['action']=torch.zeros_like(check['action']);model.eval()
   with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model.generate(check)
   assert prediction.shape==(1,30,32) and torch.isfinite(prediction).all()
   write(out/'batch_smoke.json',dict(state_shape=list(cpu['state'].shape),action_shape=list(prediction.shape),finite=True,checkpoint_compatible=True))
   del check,prediction;model.train();model.vlm.eval()
  status('READY',initialization_verified=True)
  with (out/'training_steps.jsonl').open('x') as log:
   for cursor,row in enumerate(rows):
    if row['step']<=step:continue
    with preserve_rng():cpu=inputs.get(cursor)
    batch={k:v.to('cuda:0') for k,v in cpu.items()};optimizer.zero_grad(set_to_none=True);tic=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):losses=model(batch,return_loss=True)
    loss=losses['loss'];assert torch.isfinite(loss);assert float(losses['loss_freq'].detach())==0.
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);optimizer.step();scheduler.step();torch.cuda.synchronize();step=row['step']
    item=dict(step=step,source=row['source'],window_id=row['window_id'],flow_loss=float(loss.detach()),gradient_norm=float(norm.detach()),optimizer_state_entries=len(optimizer.state),lr=optimizer.param_groups[0]['lr'],seconds=time.monotonic()-tic,real_optimizer_update=True)
    log.write(json.dumps(item,allow_nan=False)+'\n')
    if step<=5 or step%25==0:
     log.flush();os.fsync(log.fileno());difference=float((model.action_output_layer.layers[2].weight.detach().cpu()-initial).abs().max());assert difference>0 and len(optimizer.state)>0
     status('RUNNING',latest_update=item,output_weight_max_absolute_update=difference,real_optimizer_update_verified=True);print(json.dumps(item),flush=True)
    del batch,losses,loss,norm
    if step in CONFIG['checkpoint_steps']:
     optimizer.zero_grad(set_to_none=True);save_trainable_checkpoint(model,out/f'checkpoints/step_{step:04d}_trainable.pt',base_checkpoint=BASE,step=step,metadata=provenance)
    if step%500==0:
     # Atomic rolling recovery file; no asynchronous work or evaluation.
     payload=dict(python_rng=random.getstate(),numpy_rng=np.random.get_state(),step=step,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),trainable_state_dict={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),provenance=provenance)
     temp=out/'checkpoints/latest_resume.pt.tmp';torch.save(payload,temp);temp.replace(out/'checkpoints/latest_resume.pt');del payload
   assert frozen_parameter_sha256(model)==frozen and all(p.grad is None for p in model.vlm.parameters());protection_check();status('COMPLETE',post_training_evaluation=False)
 except BaseException as e:status('FAILED',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());raise
 finally:oldlock.close();lock.close()

if __name__=='__main__':
 if args.run:run()
 else:parser.error('--run required')
