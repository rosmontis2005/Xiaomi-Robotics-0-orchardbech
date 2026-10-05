"""Round1: S8 fresh B warm-start versus J8/J16 direct integrated continuation."""
import bootstrap
import argparse,fcntl,json,os,random,time,traceback,gc
from contextlib import contextmanager
from collections import Counter
import numpy as np
import torch
from recovery_common import *
from checkpoint_io import load_trainable_overlay,save_trainable_checkpoint,frozen_parameter_sha256
from inputs import ScheduledInputs

CONFIG=dict(optimizer=dict(lr=1e-5,betas=[.9,.95],weight_decay=.1,eps=1e-8,foreach=False),batch=1,gradient_accumulation=1,gradient_clip_norm=1.,lr_schedule='constant',training_repeat=1,async_train=False,enable_freq=False,freeze_vlm=True,vlm_dtype='bfloat16',non_vlm_master_dtype='float32',autocast='bfloat16',euler_steps=5,loss='original full30 active7 normalized flow, coefficient0.5',seed=42)
@contextmanager
def preserve_rng():
    py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:yield
    finally:
        random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)

def construct_model(arm):
    from mibot.models import MIMODEL
    from mibot.utils.orchard_checkpoint import load_weights
    model=MIMODEL.build(dict(type='XR0',vlm_config_path=str(PROCESSOR/'config.json'),training_repeat=1,enable_freq=False,async_train=False))
    for n,p in model.named_parameters():
        if not n.startswith('vlm.'):p.data=p.data.float()
    base_report=load_weights(model,str(BASE));model.vlm.requires_grad_(False);model.vlm.eval()
    overlay=None
    if arm=='S':overlay=load_trainable_overlay(model,ENDPOINT,base_checkpoint=BASE)
    assert all(p.dtype==torch.float32 for n,p in model.named_parameters() if not n.startswith('vlm.'))
    assert all(p.dtype==torch.bfloat16 and not p.requires_grad for p in model.vlm.parameters())
    assert model.num_steps==5 and model.flow_sampling=='beta' and model.training_repeat==1 and not model.async_train and model.freq_coefficient==0
    return model,dict(base=base_report,overlay=overlay,initialization='full10k -> B8000' if arm=='S' else 'full10k direct')

def run(args):
    out=ROOT/'round1'/args.arm/'training';out.mkdir(parents=True,exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    if (out/'run_manifest.json').exists() and not args.resume:raise FileExistsError('Already initialized; explicit resume required')
    lock=(XR0/'test_recovery_R_1004/.gpu_training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    target=8000 if args.arm=='S' else 16000;step=0;started=time.monotonic()
    def status(phase,**extra):write(out/'status.json',dict(status=phase,pid=os.getpid(),completed_updates=step,target_updates=target,time=time.time(),elapsed_seconds=time.monotonic()-started,**extra))
    try:
        protection_check();frozen=json.loads((ROOT/'frozen.json').read_text());assert frozen['status']=='FROZEN'
        for path,digest in {**frozen['files'],**frozen['assets']}.items():assert sha(path)==digest,path
        assert sha(BASE)==frozen['base_checkpoint_sha256'] and sha(ENDPOINT)==frozen['B_checkpoint_sha256']
        A=[json.loads(l) for l in (ROOT/'schedule_D_A.jsonl').read_text().splitlines()];B=[json.loads(l) for l in (ROOT/'schedule_D_B.jsonl').read_text().splitlines()]
        rows=A+([dict(r,step=r['step']+8000) for r in B] if args.arm=='J' else [])
        assert [r['step'] for r in rows]==list(range(1,target+1))
        torch.set_num_threads(1);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        status('INITIALIZING');model,load_report=construct_model(args.arm);frozen_vlm=frozen_parameter_sha256(model);model.to('cuda:0');model.train();model.vlm.eval()
        from mibot.models.runner.base_runner import BaseRunner
        options=dict(CONFIG['optimizer']);options['betas']=tuple(options['betas']);optimizer=BaseRunner.build_optimizer(dict(type='torch.optim.AdamW',params=options),((f'model.{n}',p) for n,p in model.named_parameters()))
        assert len(optimizer.state)==0
        scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer,factor=1.,total_iters=1);trainable=[p for p in model.parameters() if p.requires_grad]
        provenance=dict(initialization=load_report,base_sha256=frozen['base_checkpoint_sha256'],B_sha256=frozen['B_checkpoint_sha256'] if args.arm=='S' else None,dataset_sha256=sha(ROOT/'dataset_manifest.jsonl'),frozen_sha256=sha(ROOT/'frozen.json'),schedule_sha256={name:sha(ROOT/f'schedule_{name}.jsonl') for name in ['D_A','D_B']},stats_sha256=sha(STATS),config=CONFIG,arm=args.arm,round=1,source_counts=dict(Counter(r['source'] for r in rows)),source_sha256={str(ROOT/name):sha(ROOT/name) for name in ['train_integrated.py','inputs.py']})
        if args.resume:
            resume=torch.load(args.resume,map_location='cpu',weights_only=False,mmap=True);step=resume['step'];assert resume['provenance']==provenance
            with torch.no_grad():
                for n,v in resume['trainable_state_dict'].items():dict(model.named_parameters())[n].copy_(v)
            optimizer.load_state_dict(resume['optimizer']);scheduler.load_state_dict(resume['scheduler']);torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng']);random.setstate(resume['python_rng']);np.random.set_state(resume['numpy_rng']);del resume
        else:write(out/'run_manifest.json',dict(configuration=CONFIG,provenance=provenance,fresh_optimizer_state_entries=0,initial_load=load_report,checkpoint_steps=[8000] if args.arm=='S' else [8000,16000],J8_to_J16='same optimizer and RNG, no fresh restart'))
        with preserve_rng():inputs=ScheduledInputs(rows)
        # One real batch per source is sufficient; current-input shape is checked by loader.
        with preserve_rng(),torch.inference_mode():
            checks={}
            for source in ['original','existing_R','live_late']:
                cpu=inputs.get(next(i for i,r in enumerate(rows) if r['source']==source));assert torch.isfinite(cpu['action']).all() and torch.all(cpu['action'][:,:,7:]==0)
                checks[source]={k:list(v.shape) for k,v in cpu.items()}
            check={k:v.to('cuda:0') for k,v in cpu.items()};check['action']=torch.zeros_like(check['action']);model.eval()
            with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model.generate(check)
            assert prediction.shape==(1,30,32) and torch.isfinite(prediction).all();write(out/'batch_smoke.json',dict(checks=checks,finite=True,current_only=True))
            del prediction,check;model.train();model.vlm.eval()
        logpath=out/'training_steps.jsonl'
        if args.resume and logpath.exists():
            # Preserve failed post-resume-checkpoint updates, do not count them twice.
            records=[json.loads(l) for l in logpath.read_text().splitlines()];append(out/'failed_updates.jsonl',dict(resume_step=step,discarded=[r for r in records if r['step']>step]));logpath.write_text(''.join(json.dumps(r)+'\n' for r in records if r['step']<=step))
        with logpath.open('a') as log:
            for cursor,row in enumerate(rows):
                if row['step']<=step:continue
                with preserve_rng():cpu=inputs.get(cursor)
                batch={k:v.to('cuda:0') for k,v in cpu.items()};optimizer.zero_grad(set_to_none=True);tic=time.monotonic()
                with torch.autocast('cuda',dtype=torch.bfloat16):losses=model(batch,return_loss=True)
                loss=losses['loss'];assert torch.isfinite(loss) and float(losses['loss_freq'].detach())==0.
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);optimizer.step();scheduler.step();torch.cuda.synchronize();step=row['step']
                item=dict(step=step,source=row['source'],group=row['group'],window_id=row['window_id'],flow_loss=float(loss.detach()),gradient_norm=float(norm.detach()),lr=optimizer.param_groups[0]['lr'],seconds=time.monotonic()-tic,real_optimizer_update=True)
                log.write(json.dumps(item,allow_nan=False)+'\n')
                if step<=5 or step%25==0:log.flush();status('RUNNING',latest_update=item);print(json.dumps(item),flush=True)
                del batch,losses,loss,norm
                if step in [8000,16000]:
                    optimizer.zero_grad(set_to_none=True);save_trainable_checkpoint(model,out/f'checkpoints/step_{step:04d}_trainable.pt',base_checkpoint=BASE,step=step,metadata=provenance)
                if step%500==0:
                    payload=dict(python_rng=random.getstate(),numpy_rng=np.random.get_state(),step=step,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),trainable_state_dict={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),provenance=provenance)
                    temp=out/'checkpoints/latest_resume.pt.tmp';torch.save(payload,temp);temp.replace(out/'checkpoints/latest_resume.pt');del payload
        assert frozen_parameter_sha256(model)==frozen_vlm and all(p.grad is None for p in model.vlm.parameters());protection_check();status('COMPLETE')
        records=[json.loads(l) for l in logpath.read_text().splitlines()];assert len(records)==target
        write(out/'training_summary.json',dict(arm=args.arm,updates=target,initialization=load_report,optimizer=CONFIG['optimizer'],fresh_optimizer=True,source_counts=dict(Counter(r['source'] for r in records)),loss_by_source={s:dict(first500=float(np.mean([r['flow_loss'] for r in records[:500] if r['source']==s])),last500=float(np.mean([r['flow_loss'] for r in records[-500:] if r['source']==s]))) for s in ['original','existing_R','live_late']},checkpoints={str(p):sha(p) for p in (out/'checkpoints').glob('*trainable.pt')},frozen_vlm_unchanged=True))
    except BaseException as e:status('FAILED',error=str(e),traceback=traceback.format_exc());raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['S','J'],required=True);p.add_argument('--resume');run(p.parse_args())
