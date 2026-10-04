#!/usr/bin/env python3
"""Isolated A/B training. ROOT schedules GPU; this file never launches the other arm.

--check-prepared is CPU only. --arm B --run starts one arm; --resume restores
model, Adam moments, scheduler, Python/numpy/CPU/CUDA RNG and schedule cursor.
"""
import bootstrap
import argparse
from collections import Counter, OrderedDict
from datetime import datetime, timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from torch.utils.data import Subset
import ab_common as common
from ab_common import ROOT, XR0, ORCHARD, DATA, STATS, BASE, PROCESSOR, SELECTION, CACHE
from checkpoint_io import file_sha256, frozen_parameter_sha256, save_trainable_checkpoint, load_trainable_overlay

CONFIG_PATH=ROOT/'config_ab.json'
RESUME_FORMAT='orchard_ab_exact_resume_v1'


def utc_now():return datetime.now(timezone.utc).isoformat()

def configuration():return json.loads(CONFIG_PATH.read_text())

def source_hashes():return {str(p):file_sha256(p) for p in common.SOURCE_FILES}

def schedule_path(arm):return ROOT/'schedules'/f'{arm}.jsonl'

def load_schedule(arm):
    rows=[json.loads(line) for line in schedule_path(arm).read_text().splitlines() if line.strip()]
    assert len(rows)==8000 and [int(r['step']) for r in rows]==list(range(1,8001))
    return rows

def ensure_precision_ready():
    path=ROOT/'precision_fix.json';report=json.loads(path.read_text())
    assert report['status']=='PASS' and report['protected_files_unchanged'] is True and report['gpu_released'] is True
    source=Path(report['production_fix']['source'])
    assert source.resolve()==(XR0/'mibot/utils/orchard_checkpoint.py').resolve()
    assert file_sha256(source)==report['production_fix']['after_sha256'],'Precision loader hash mismatch'
    return report


def check_prepared():
    path=ROOT/'prepared_checks.json';report=json.loads(path.read_text())
    assert report['status']=='PASS'
    for name,digest in report['artifact_sha256'].items():
        assert file_sha256(name)==digest,f'Prepared artifact changed: {name}'
    assert source_hashes()==report['source_sha256'],'Source changed since CPU preparation'
    assert file_sha256(BASE)==report['base_sha256']
    cache=torch.load(CACHE,map_location='cpu',weights_only=True,mmap=True)
    assert len(cache['samples'])==240
    counts=Counter(s['meta']['evaluation_set'] for s in cache['samples'])
    assert dict(counts)==configuration()['evaluation_sets']
    assert all(s['batch']['action'].dtype==torch.float32 for s in cache['samples'])
    return report,cache


def capture_rng():
    ns=np.random.get_state()
    return dict(python=random.getstate(),numpy=dict(algorithm=ns[0],keys=torch.from_numpy(ns[1].astype(np.int64)),position=ns[2],has_gauss=ns[3],cached_gaussian=ns[4]),
                torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n['algorithm'],n['keys'].numpy().astype(np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:torch.cuda.set_rng_state_all(state['torch_cuda'])


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def active_state(model):return {n:p.detach().cpu().float().clone() for n,p in model.named_parameters() if not n.startswith('vlm.')}


def restore_active(model,state):
    params={n:p for n,p in model.named_parameters() if not n.startswith('vlm.')}
    assert set(params)==set(state),'Incomplete resume trainable keys'
    with torch.no_grad():
        for name,value in state.items():
            assert params[name].dtype==torch.float32 and value.dtype==torch.float32 and params[name].shape==value.shape
            assert torch.isfinite(value).all(),name
            params[name].copy_(value)


def atomic_torch_save(path,payload):
    path=Path(path).resolve();assert path.is_relative_to(ROOT)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.pending_{os.getpid()}_{time.time_ns()}')
    with temp.open('xb') as out:
        torch.save(payload,out);out.flush();os.fsync(out.fileno())
    os.replace(temp,path)


def save_resume(path,model,optimizer,scheduler,state,provenance):
    payload=dict(format=RESUME_FORMAT,trainable_state_dict=active_state(model),optimizer=cpu_tree(optimizer.state_dict()),
                 scheduler=scheduler.state_dict(),rng=capture_rng(),state=state,provenance=provenance)
    atomic_torch_save(path,payload)
    return dict(path=str(path),step=state['step'],bytes=path.stat().st_size,written_utc=utc_now())


def restore_resume(path,model,optimizer,scheduler,provenance):
    payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    assert payload['format']==RESUME_FORMAT and payload['provenance']==provenance
    restore_active(model,payload['trainable_state_dict'])
    optimizer.load_state_dict(cpu_tree(payload['optimizer']));scheduler.load_state_dict(payload['scheduler'])
    restore_rng(payload['rng'])
    return payload['state']


class ScheduledInputs:
    """Full original Dataset first; frozen schedule is a Subset of its valid indices."""
    def __init__(self,schedule,limit=256):
        from mibot.data.datasets.orchardbench_dataset import OrchardBenchDataset
        from mibot.data.datamodule.orchardbench_datamodule import OrchardBenchDataModule
        self.dataset=OrchardBenchDataset(dict(train_datasets=dict(root=str(DATA),split='train',episode_ids=None,
                                      stats_path=str(STATS),action_length=30,batch_size=1)))
        lookup={(str(Path(p).resolve()),int(f)):i for i,(p,f) in enumerate(self.dataset.samples)}
        self.indices=[lookup[(str(Path(row['annotation']).resolve()),int(row['frame']))] for row in schedule]
        self.subset=Subset(self.dataset,self.indices)
        self.dm=OrchardBenchDataModule(dict(processor_path=str(PROCESSOR)))
        self.cache=OrderedDict();self.limit=limit;self.hits=0;self.misses=0
    def get(self,cursor):
        key=self.indices[cursor]
        if key in self.cache:
            self.hits+=1;self.cache.move_to_end(key);return self.cache[key]
        batch=self.dm.collate_fn([self.subset[cursor]])
        assert all(isinstance(v,torch.Tensor) and v.device.type=='cpu' for v in batch.values())
        assert batch['action'].dtype==torch.float32 and tuple(batch['action'].shape)==(1,30,32)
        self.cache[key]={k:v.contiguous() for k,v in batch.items()};self.misses+=1
        if len(self.cache)>self.limit:self.cache.popitem(last=False)
        return self.cache[key]


def construct_model():
    """Cast destination before strict source load; never round source FP32 via BF16."""
    from mibot.models import MIMODEL
    from mibot.utils.orchard_checkpoint import load_weights
    model=MIMODEL.build(dict(type='XR0',vlm_config_path=str(PROCESSOR/'config.json'),training_repeat=1,enable_freq=False,async_train=False))
    for name,param in model.named_parameters():
        if not name.startswith('vlm.'):param.data=param.data.float()
    load_report=load_weights(model,str(BASE))
    model.vlm.requires_grad_(False);model.vlm.eval()
    assert all(p.dtype==torch.float32 and p.requires_grad for n,p in model.named_parameters() if not n.startswith('vlm.'))
    payload=torch.load(BASE,map_location='cpu',weights_only=True,mmap=True)
    raw=payload.get('module',payload.get('state_dict',payload));source={k.removeprefix('model.'):v for k,v in raw.items()}
    count=0
    for name,param in model.named_parameters():
        if name.startswith('vlm.'):continue
        assert source[name].dtype==torch.float32 and torch.equal(param.detach().cpu(),source[name]),f'Source FP32 lost: {name}'
        count+=1
    del payload,raw,source;gc.collect()
    assert model.num_steps==5 and model.flow_sampling=='beta' and model.training_repeat==1 and not model.async_train
    return model,dict(**load_report,non_vlm_source_fp32_tensor_exact_equal=True,source_fp32_parameter_tensors=count,
        historical_note='Unlike old M0 runner initialization, load original source FP32 directly into FP32 destinations; no initial BF16 roundtrip.')


def promote_link(source,target):
    target=Path(target);temp=target.with_name(target.name+f'.pending_{time.time_ns()}')
    os.link(source,temp);os.replace(temp,target)


def repair_log(path,step):
    if not path.exists():return
    lines=path.read_text().splitlines();keep=[];discard=[]
    for line in lines:
        try:record=json.loads(line)
        except json.JSONDecodeError:discard.append(line);continue
        (keep if record['step']<=step else discard).append(line)
    assert [json.loads(line)['step'] for line in keep]==list(range(1,step+1)), 'Missing committed update log'
    if discard:
        archive=path.with_name(f'interrupted_uncommitted_steps_{time.time_ns()}.jsonl');archive.write_text('\n'.join(discard)+'\n')
        temp=path.with_name(path.name+'.pending_repair');temp.write_text('\n'.join(keep)+('\n' if keep else ''));os.replace(temp,path)


def run(arm,resume=False):
    cfg=configuration();prepared,cache=check_prepared();precision=ensure_precision_ready();schedule=load_schedule(arm)
    out=ROOT/'arms'/arm;out.mkdir(parents=True,exist_ok=True)
    global_lock=(ROOT/'.gpu_training.lock').open('a+');fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=(out/'.process.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=out/'status.json';manifest_path=out/'run_manifest.json';resume_path=out/'checkpoints/latest_resume.pt'
    if not resume and manifest_path.exists():raise FileExistsError('Existing arm; explicitly use --resume')
    if resume and not resume_path.exists():raise FileNotFoundError('No committed resume state')
    process_started=time.monotonic();step=0
    def status(phase,**extra):
        common.write_json(status_path,dict(arm=arm,status=phase,pid=os.getpid(),completed_updates=step,
                    target_updates=8000,last_heartbeat_utc=utc_now(),process_elapsed_s=time.monotonic()-process_started,**extra),replace=True)
    status('INITIALIZING',resumed=resume)
    try:
        if not torch.cuda.is_available():raise RuntimeError('Root-scheduled CUDA required')
        torch.set_num_threads(1);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        model,load_report=construct_model();frozen=frozen_parameter_sha256(model)
        initial_output=model.action_output_layer.layers[2].weight.detach().cpu().clone()
        device=torch.device('cuda:0');model.to(device);model.train();model.vlm.eval()
        from mibot.models.runner.base_runner import BaseRunner
        op=dict(cfg['optimizer']);op.pop('type');op['betas']=tuple(op['betas'])
        optimizer=BaseRunner.build_optimizer(dict(type='torch.optim.AdamW',params=op),((f'model.{n}',p) for n,p in model.named_parameters()))
        scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer,factor=1.,total_iters=1)
        assert len(optimizer.state)==0
        provenance=dict(arm=arm,base_checkpoint=str(BASE),base_sha256=file_sha256(BASE),selection_sha256=file_sha256(SELECTION),
            schedule_sha256=file_sha256(schedule_path(arm)),stats_sha256=file_sha256(STATS),eval_cache_sha256=file_sha256(CACHE),
            prepared_checks_sha256=file_sha256(ROOT/'prepared_checks.json'),config_sha256=file_sha256(CONFIG_PATH),precision_fix_sha256=file_sha256(ROOT/'precision_fix.json'),
            source_sha256=source_hashes(),frozen_vlm_parameter_sha256=frozen)
        state=dict(step=0,window_counts={},episode_counts={},group_counts={},training_seconds=0.,data_seconds=0.,evaluation_seconds=0.,
                   evaluated_steps=[],evaluations=[],best_rank=None,best_step=None,checkpoint_records={},resume_history=[])
        if resume:
            existing=json.loads(manifest_path.read_text());assert existing['provenance']==provenance
            state=restore_resume(resume_path,model,optimizer,scheduler,provenance);step=state['step']
            state['resume_history'].append(dict(utc=utc_now(),from_step=step,pid=os.getpid()))
            repair_log(out/'training_steps.jsonl',step)
        else:
            random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            manifest=dict(schema='orchard_ab_run_v1',arm=arm,configuration=cfg,provenance=provenance,initial_load_report=load_report,
                initial_optimizer_state_entries=0,gpu_name=torch.cuda.get_device_name(0),torch_version=str(torch.__version__),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                train_episode_ids=sorted({r['episode_id'] for r in schedule}),schedule_rows=8000,
                evaluation_windows=[s['meta']['window_id'] for s in cache['samples']],
                best_rule=common.BEST_RULE,primary_endpoint_step=8000,created_utc=utc_now(),
                resume_contract='Atomic rolling optimizer/model/RNG/cursor every500, plus evaluation boundaries; at most499 uncommitted updates replayed after interruption.')
            common.write_json(manifest_path,manifest)
            record=save_resume(resume_path,model,optimizer,scheduler,state,provenance)
            status('READY',latest_resume=record)
        trainable=[p for p in model.parameters() if p.requires_grad]
        samples=cache['samples'];mean=cache['mean'].numpy();std=cache['std'].numpy()
        metadata=dict(provenance=provenance,configuration=cfg,best_rule=common.BEST_RULE)
        def eval_and_checkpoint():
            nonlocal state
            if step in state['evaluated_steps']:return
            optimizer.zero_grad(set_to_none=True)
            status('EVALUATING',evaluation_step=step,evaluation_windows_complete=0,evaluation_windows_total=len(samples))
            result=common.evaluate(model,samples,mean,std,step,device,out,
                status_callback=lambda done,total:status('EVALUATING',evaluation_step=step,evaluation_windows_complete=done,evaluation_windows_total=total))
            path=out/'checkpoints'/f'step_{step:04d}_trainable.pt'
            if path.exists():
                payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                assert payload['step']==step and payload['metadata']['provenance']==provenance
                record=dict(path=str(path),step=step,checkpoint_sha256=file_sha256(path),bytes=path.stat().st_size)
                del payload
            else:
                stale=path.with_name(path.name+'.tmp')
                if stale.exists():os.replace(stale,stale.with_name(stale.name+f'.interrupted_{time.time_ns()}'))
                record=save_trainable_checkpoint(model,path,base_checkpoint=BASE,step=step,metadata={**metadata,'fit_gate':result['fit_gate']})
            rank=result['fit_gate']['ranking'];state['checkpoint_records'][str(step)]=record
            if state['best_rank'] is None or tuple(rank)<tuple(state['best_rank']):
                state['best_rank']=rank;state['best_step']=step;promote_link(path,out/'checkpoints/best_trainable.pt')
                common.write_json(out/'best_checkpoint.json',{**record,'fit_gate':result['fit_gate'],'rule':common.BEST_RULE},replace=True)
            state['evaluated_steps'].append(step);state['evaluation_seconds']+=result['elapsed_s']
            state['evaluations'].append(dict(step=step,report=str(out/'evaluations'/f'step_{step:04d}'/'report.json'),fit_gate=result['fit_gate']))
            save_resume(resume_path,model,optimizer,scheduler,state,provenance)
            status('RUNNING',best_step=state['best_step'],last_evaluation_step=step,latest_resume_step=step)
        if step in cfg['evaluation_steps']:eval_and_checkpoint()
        with common.preserve_rng():inputs=ScheduledInputs(schedule,cfg['cpu_input_lru_windows'])
        assert inputs.dataset.stats['source_sha256']==prepared['full_train_stats_fingerprint']
        logfile=out/'training_steps.jsonl'
        with logfile.open('a' if resume else 'x') as log:
            for cursor in range(step,8000):
                row=schedule[cursor];step=cursor+1;assert int(row['step'])==step
                tic=time.monotonic()
                with common.preserve_rng():cpu_batch=inputs.get(cursor)
                data_s=time.monotonic()-tic
                optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();tic=time.monotonic()
                batch=common.to_device(cpu_batch,device)
                with torch.autocast('cuda',dtype=torch.bfloat16):losses=model(batch,return_loss=True)
                loss=losses['loss']
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at{step}')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,cfg['gradient_clip_norm'],error_if_nonfinite=True)
                optimizer.step();scheduler.step();torch.cuda.synchronize();seconds=time.monotonic()-tic
                state['step']=step;state['training_seconds']+=seconds;state['data_seconds']+=data_s
                for key,field in (('window_counts','window_id'),('episode_counts','episode_id'),('group_counts','group')):
                    value=str(row[field]);state[key][value]=state[key].get(value,0)+1
                item=dict(step=step,arm=arm,window_id=row['window_id'],episode_id=row['episode_id'],frame=int(row['frame']),group=row['group'],split='train',
                    flow_loss=float(loss.detach()),flow_loss_mse=float(losses['loss_mse'].detach()),flow_loss_freq=float(losses['loss_freq'].detach()),
                    gradient_norm_before_clip=float(norm.detach()),gradients_finite=True,lr=optimizer.param_groups[0]['lr'],
                    step_seconds=seconds,data_seconds=data_s,optimizer_state_entries=len(optimizer.state))
                log.write(json.dumps(item,allow_nan=False)+'\n')
                del batch,losses,loss,norm
                if step%cfg['status_every_updates']==0:
                    log.flush();os.fsync(log.fileno())
                    status('RUNNING',latest_loss=item['flow_loss'],gradients_finite=True,best_step=state['best_step'],
                        observed_episode_count=len(state['episode_counts']),data_cache_hits=inputs.hits,data_cache_misses=inputs.misses,
                        latest_resume_step=(step//cfg['resume_every_updates'])*cfg['resume_every_updates'] if step%cfg['resume_every_updates'] else step-cfg['resume_every_updates'])
                    print(json.dumps(dict(event='train_progress',**item)),flush=True)
                if step%cfg['resume_every_updates']==0:
                    optimizer.zero_grad(set_to_none=True);record=save_resume(resume_path,model,optimizer,scheduler,state,provenance)
                    status('RUNNING',latest_resume=record,best_step=state['best_step'])
                if step in cfg['evaluation_steps']:eval_and_checkpoint()
        optimizer.zero_grad(set_to_none=True)
        final_frozen=frozen_parameter_sha256(model);assert final_frozen==frozen
        assert all(not p.requires_grad and p.grad is None for p in model.vlm.parameters())
        assert sum(state['window_counts'].values())==8000 and len(state['episode_counts'])==128
        assert all(p.dtype==torch.float32 for p in trainable)
        assert source_hashes()==provenance['source_sha256']
        update=float((model.action_output_layer.layers[2].weight.detach().cpu()-initial_output).abs().max());assert update>0
        final=out/'checkpoints/step_8000_trainable.pt';promote_link(final,out/'checkpoints/final_trainable.pt')
        overlay_report=load_trainable_overlay(model,final,base_checkpoint=BASE)
        summary=dict(status='COMPLETE',arm=arm,completed_updates=step,val_optimizer_updates=0,
            unique_train_windows_seen=len(state['window_counts']),train_episodes_seen=len(state['episode_counts']),
            per_window_counts=state['window_counts'],per_episode_counts=state['episode_counts'],group_counts=state['group_counts'],
            frozen_vlm_unchanged=True,frozen_vlm_parameter_sha256_before=frozen,frozen_vlm_parameter_sha256_after=final_frozen,
            trainable_master_dtype='float32',output_weight_max_absolute_update=update,all_logged_gradients_finite=True,
            training_seconds=state['training_seconds'],data_seconds=state['data_seconds'],evaluation_seconds=state['evaluation_seconds'],
            process_elapsed_s=time.monotonic()-process_started,resume_history=state['resume_history'],best_step=state['best_step'],
            primary_endpoint_step=8000,checkpoint_records=state['checkpoint_records'],final_overlay_reload=overlay_report,evaluations=state['evaluations'],
            source_sha256=source_hashes(),provenance=provenance,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        common.write_json(out/'training_summary.json',summary,replace=True)
        status('COMPLETE',best_step=state['best_step'],summary=str(out/'training_summary.json'))
        print(json.dumps(dict(event='training_complete',arm=arm,summary=str(out/'training_summary.json'))),flush=True)
    except BaseException as exc:
        status('FAILED',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close();fcntl.flock(global_lock,fcntl.LOCK_UN);global_lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=('A','B'),default='B')
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--check-prepared',action='store_true');mode.add_argument('--run',action='store_true')
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.check_prepared:
        report,cache=check_prepared();assert not torch.cuda.is_initialized()
        print(json.dumps(dict(status='PASS',evaluation_windows=len(cache['samples']),cuda_initialized=False)))
    else:run(a.arm,a.resume)

if __name__=='__main__':main()
