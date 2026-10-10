from common import *
import argparse,random,time,gc,signal,traceback,fcntl
from contextlib import contextmanager
from collections import deque
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from dataset import V2Dataset,collator

KEY_STEPS=[0,2000,5000,10000,20000,40000,60000]
def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(r):random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state_all(r['cuda'])
@contextmanager
def preserve_rng():
 r=rng_state()
 try:yield
 finally:restore_rng(r)
def active(model):return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if not n.startswith('vlm.')}
def overlay(model,state):
 params={n:p for n,p in model.named_parameters() if not n.startswith('vlm.')};assert params.keys()==state.keys()
 with torch.no_grad():
  for n,p in params.items():assert p.dtype==torch.float32 and state[n].dtype==torch.float32;p.copy_(state[n])
def atomic_save(path,payload):
 temp=path.with_suffix('.tmp');torch.save(payload,temp)
 with temp.open('rb') as f:os.fsync(f.fileno())
 temp.replace(path)
def construct():
 from mibot.models import MIMODEL
 from mibot.utils.orchard_checkpoint import load_weights
 model=MIMODEL.build(dict(type='XR0',vlm_config_path=str(BASE/'config.json'),training_repeat=1,enable_freq=False,async_train=False))
 report=load_weights(model,str(BASE));model.vlm.requires_grad_(False);model.vlm.eval()
 assert all(p.dtype==torch.float32 and p.requires_grad for n,p in model.named_parameters() if not n.startswith('vlm.'))
 assert all(p.dtype==torch.bfloat16 and not p.requires_grad for p in model.vlm.parameters())
 # Confirm destination values against original shards, before moving to GPU.
 from safetensors import safe_open
 index=json.loads((BASE/'model.safetensors.index.json').read_text())['weight_map'];params=dict(model.named_parameters());checked=0
 for shard in set(index.values()):
  with safe_open(str(BASE/shard),framework='pt') as f:
   for n in f.keys():
    if not n.startswith('vlm.'):
     assert torch.equal(params[n].detach(),f.get_tensor(n).float()),n;checked+=1
 report['non_vlm_exact_source_tensors']=checked
 return model,report

def gpu(batch):return {k:v.to('cuda') for k,v in batch.items()}
def fixed_eval(model,ds,collate,step,generate):
 path=ROOT/'evaluations'/f'step_{step:06d}.json';records=[];started=time.monotonic()
 with preserve_rng(),torch.no_grad():
  for i,row in enumerate(ds.rows):
   batch=gpu(collate([ds[i]]));seed(row['eval_seed'])
   # XR0 model.eval() changes flow training into action-generation MSE.
   # Keep original training branch for the SAME flow loss; VLM stays eval.
   model.train();model.vlm.eval()
   with torch.autocast('cuda',dtype=torch.bfloat16):losses=model(dict(batch),return_loss=True)
   record=dict(**row,loss=float(losses['loss']),loss_mse=float(losses['loss_mse']),loss_freq=float(losses['loss_freq']))
   if generate:
    model.eval();seed(row['eval_seed']);batch['action']=torch.zeros_like(batch['action'])
    with torch.autocast('cuda',dtype=torch.bfloat16):pred=model.generate(dict(batch))[0].float().cpu().numpy()
    physical=pred*(ds.std+1e-6)+ds.mean
    from dataset import arrays
    target=arrays(row['seed'])['actions'][row['index']]
    pe=np.linalg.norm(physical[:,:3]-target[:,:3],axis=1)*1000
    re=(Rotation.from_rotvec(physical[:,3:6]).inv()*Rotation.from_rotvec(target[:,3:6])).magnitude()
    we=np.abs(physical[:,6]-target[:,6])*1000
    record['errors']={name:dict(position_mm=float(pe[a:b].mean()),rotation_rad=float(re[a:b].mean()),width_mm=float(we[a:b].mean())) for name,a,b in [('h1:5',0,5),('h6:15',5,15),('h16:30',15,30),('full30',0,30)]}
    record['target_phase_errors']={phase:{name:dict(position_mm=float(pe[a:b][mask[a:b]].mean()),rotation_rad=float(re[a:b][mask[a:b]].mean()),width_mm=float(we[a:b][mask[a:b]].mean()),targets=int(mask[a:b].sum())) for name,a,b in [('h1:5',0,5),('h6:15',5,15),('h16:30',15,30),('full30',0,30)] if mask[a:b].any()} for j,phase in enumerate(PHASES) if (mask:=arrays(row['seed'])['phases'][row['frame']:row['frame']+30]==j).any()}
   records.append(record)
   if i%64==0:print('PANEL',step,i,flush=True)
  model.train();model.vlm.eval()
 result=dict(step=step,flow_loss={s:float(np.mean([r['loss'] for r in records if r['split']==s])) for s in ['train','validation']},records=records,elapsed=time.monotonic()-started,fixed_noise='Reset Python/numpy/CPU/CUDA RNG per window; preserve outer training RNG; original training branch with no_grad and frozen eval VLM; same hardware kernels may have residual nondeterministic rounding.')
 result['gap']=result['flow_loss']['validation']-result['flow_loss']['train'];write(path,result);return result

def run(resume=False):
 torch.set_num_threads(1);seed(42);cfg=json.loads((ROOT/'config.json').read_text());frozen=json.loads((ROOT/'frozen.json').read_text())
 for p,h in frozen['artifacts'].items():assert sha(p)==h,p
 lock=(ROOT/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 checkpoints=ROOT/'checkpoints';checkpoints.mkdir(exist_ok=True);latest=checkpoints/'latest_resume.pt';logpath=ROOT/'training_steps.jsonl';step=0;stop=False
 def status(phase,**kw):write(ROOT/'status.json',dict(phase=phase,step=step,budget=60000,pid=os.getpid(),time=time.time(),**kw))
 def request_stop(*_):
  nonlocal stop;stop=True
 signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
 status('INITIALIZING');model,report=construct();model.to('cuda');model.train();model.vlm.eval()
 from mibot.models.runner.base_runner import BaseRunner
 opt=BaseRunner.build_optimizer(dict(type='torch.optim.AdamW',params=dict(lr=1e-5,betas=(.9,.95),weight_decay=.1,eps=1e-8,foreach=False)),model.named_parameters());scheduler=torch.optim.lr_scheduler.ConstantLR(opt,factor=1.,total_iters=1)
 params=[p for p in model.parameters() if p.requires_grad];schedule=lines(ROOT/'schedule.jsonl');ds=V2Dataset(schedule,'train');panel=V2Dataset(lines(ROOT/'panels.jsonl'));collate=collator();elapsed=0.;evaluated=[]
 def checkpoint():
  atomic_save(latest,dict(step=step,trainable=active(model),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(),frozen=frozen,elapsed=elapsed,evaluated=evaluated))
 def endpoint():
  path=checkpoints/f'step_{step:06d}.pt'
  if not path.exists():atomic_save(path,dict(step=step,trainable=active(model),frozen=frozen,base=str(BASE)))
 def evaluate():
  if step not in evaluated:
   opt.zero_grad(set_to_none=True);status('FIXED_PANEL');fixed_eval(model,panel,collate,step,step in KEY_STEPS);evaluated.append(step);checkpoint()
 if resume:
  payload=torch.load(latest,map_location='cpu',weights_only=False);assert payload['frozen']==frozen;overlay(model,payload['trainable']);opt.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler']);restore_rng(payload['rng']);step=payload['step'];elapsed=payload['elapsed'];evaluated=payload['evaluated'];del payload;gc.collect()
  rows=lines(logpath);committed=[r for r in rows if r['step']<=step];discard=[r for r in rows if r['step']>step]
  assert [r['step'] for r in committed]==list(range(1,step+1))
  if discard:jsonl(ROOT/f'interrupted_uncommitted_{time.time_ns()}.jsonl',discard)
  temp=logpath.with_suffix('.tmp');temp.write_text(''.join(json.dumps(r)+'\n' for r in committed));temp.replace(logpath)
 else:
  assert not logpath.exists() and not latest.exists()
  # Exactly one real batch forward/backward, no optimizer update, RNG restored.
  from treesim.orchard_command import encode_window
  from treesim.orchard_action import decode_targets
  r=schedule[0];traj=json.loads(Path(ds.manifest[r['seed']]['annotation']).read_text());physical=encode_window(traj,r['frame']);o=traj['observations'][r['frame']];p,rot,w=decode_targets(physical,o['tcp_pos_world'],Rotation.from_quat(o['tcp_quat_world']).as_matrix());cmds=traj['commands'][r['frame']:r['frame']+30]
  assert np.allclose(p,[c['position'] for c in cmds],atol=2e-6) and np.allclose(rot,[c['rotation'] for c in cmds],atol=2e-6) and np.allclose(w,[c['width'] for c in cmds],atol=1e-7)
  with preserve_rng():
   batch=gpu(collate([ds[0]]));tic=time.monotonic()
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(batch,return_loss=True)['loss']
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);torch.cuda.synchronize()
   write(ROOT/'preflight.json',dict(initial_load=report,roundtrip=True,loss=float(loss.detach()),gradient_norm=float(norm),seconds=time.monotonic()-tic,optimizer_updates=0,optimizer_state_entries=len(opt.state),peak_allocated=torch.cuda.max_memory_allocated(),vlm_frozen=all(p.grad is None and not p.requires_grad for p in model.vlm.parameters()),trainable_parameters=sum(p.numel() for p in params)))
   opt.zero_grad(set_to_none=True);del batch,loss,norm,traj;gc.collect()
  endpoint();checkpoint()
 if step in KEY_STEPS:endpoint()
 if step%2000==0 or step in KEY_STEPS:evaluate()
 rolling=deque(maxlen=1000)
 if logpath.exists():rolling.extend(r['loss'] for r in lines(logpath)[-1000:])
 try:
  with logpath.open('a',buffering=1) as log:
   for cursor in range(step,60000):
    tic=time.monotonic();batch=gpu(collate([ds[cursor]]));torch.cuda.synchronize();load_seconds=time.monotonic()-tic
    opt.zero_grad(set_to_none=True);tic=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):losses=model(batch,return_loss=True)
    loss=losses['loss']
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    opt.step();scheduler.step();torch.cuda.synchronize();seconds=time.monotonic()-tic;step=cursor+1;elapsed+=seconds+load_seconds
    rolling.append(float(loss.detach()));row=dict(**schedule[cursor],loss=float(loss.detach()),loss_mse=float(losses['loss_mse'].detach()),loss_freq=float(losses['loss_freq'].detach()),lr=opt.param_groups[0]['lr'],gradient_norm_before_clip=float(norm),clipped=bool(norm>1),batch_loading_seconds=load_seconds,optimization_seconds=seconds,peak_allocated=torch.cuda.max_memory_allocated(),rolling={str(n):float(np.mean(list(rolling)[-n:])) for n in [50,200,1000]},actual_optimizer_update=True)
    log.write(json.dumps(row,allow_nan=False)+'\n');del batch,loss,losses,norm
    if step<=5 or step%25==0:status('TRAINING',loss=row['loss'],seconds_per_update=seconds+load_seconds,elapsed_training=elapsed);print('UPDATE',step,row['loss'],seconds,load_seconds,flush=True)
    if step%500==0 or stop:log.flush();os.fsync(log.fileno());opt.zero_grad(set_to_none=True);checkpoint()
    if step in KEY_STEPS:endpoint()
    if step%2000==0 or step in KEY_STEPS:evaluate()
    if stop:status('INTERRUPTED_RESUMABLE');return
  status('TRAINING_COMPLETE',elapsed_training=elapsed);write(ROOT/'training_complete.json',dict(step=step,elapsed=elapsed,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),vlm_grad_absent=all(p.grad is None for p in model.vlm.parameters()),checkpoints={str(p):sha(p) for p in checkpoints.glob('step_*.pt')}))
 except BaseException as e:
  write(ROOT/f'anomaly_{time.time_ns()}.json',dict(step=step,error=str(e),traceback=traceback.format_exc()));status('ENGINEERING_ERROR',error=str(e));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');run(p.parse_args().resume)
