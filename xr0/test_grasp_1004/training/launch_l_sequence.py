#!/usr/bin/env python3
"""Explicitly approved D0 -> B8000 extra reference -> L training handoff.

No polling/waiting for D0 and no automatic startup. --launch must be explicitly
called after review with all three frozen hashes; --check only inspects gates.
The worker runs reference and training in separate processes, never concurrently.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
EXPERIMENT=ROOT.parent
EVALUATION=EXPERIMENT/'evaluation'
PYTHON=EXPERIMENT.parent/'.venv-orchard/bin/python'
EXPECTED_PREPARED='bf641db8f102b0c0b63b08b9dd273f2cc5868fac58ec281c7c999b3205821a37'
EXPECTED_D0='74d71006104fe63c3a963c724adbdeed0f31db867c072662d6bfb3bc69b73eb0'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def stamp():return datetime.now(timezone.utc).isoformat()


def save(path,value,replace=False):
    path=Path(path);assert path.resolve().is_relative_to(ROOT)
    if not replace:
        with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    else:
        tmp=path.with_name(path.name+f'.pending_{os.getpid()}')
        with tmp.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
        os.replace(tmp,path)


def gpu_empty():
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True,capture_output=True,check=True)
    rows=[line.strip() for line in p.stdout.splitlines() if line.strip()]
    assert not rows,('GPU compute processes still active',rows)
    return rows


def gate(args):
    assert args.approved_prepared==EXPECTED_PREPARED==sha(ROOT/'prepared_checks.json')
    assert args.approved_d0==EXPECTED_D0==sha(EVALUATION/'protocol.json')
    assert args.approved_launcher==sha(__file__),'Launcher hash must be explicitly reviewed'
    prepared=json.loads((ROOT/'prepared_checks.json').read_text());assert prepared['status']=='PASS'
    for path,digest in {**prepared['artifact_sha256'],**prepared['source_sha256']}.items():assert sha(path)==digest,path
    cfg=json.loads((ROOT/'config_l.json').read_text());assert cfg['updates_per_arm']==8000
    protocol=json.loads((EVALUATION/'protocol.json').read_text())
    queue=json.loads((EVALUATION/'queue_status.json').read_text())
    assert queue['status']=='COMPLETE' and queue['episodes']==34,'D0 must already be complete; launcher does not wait'
    assert queue['protocol_sha256']==EXPECTED_D0
    for path,digest in protocol['source_sha256'].items():assert sha(path)==digest,path
    assert sha(protocol['checkpoint']['path'])==protocol['checkpoint']['sha256']
    jobs=[j for stage in protocol['queue_order'] for j in protocol['jobs'][stage]]
    assert len(jobs)==34 and len({j['job_id'] for j in jobs})==34
    completions={};files=0
    for job in jobs:
        scene=job['scene'];assert sha(scene['annotation'])==scene['annotation_sha256']
        folder=EVALUATION/'rollouts'/job['job_id'];path=folder/'completed.json';d=json.loads(path.read_text())
        assert d['status']=='COMPLETE' and d['job_id']==job['job_id'] and d['protocol_sha256']==EXPECTED_D0
        assert d['checkpoint_sha256']==(None if job['is_gt'] else protocol['checkpoint']['sha256'])
        assert {'summary.json','steps.jsonl'}.issubset(d['files_sha256'])
        for name,digest in d['files_sha256'].items():assert sha(folder/name)==digest;files+=1
        completions[job['job_id']]=sha(path)
    gpu_empty()
    free=shutil.disk_usage(ROOT).free
    assert free>=40*1024**3,('Need >=40GiB free for 5 overlays, atomic resume and diagnostics',free)
    return dict(status='PASS',checked_utc=stamp(),prepared_sha256=EXPECTED_PREPARED,
        D0_protocol_sha256=EXPECTED_D0,launcher_sha256=args.approved_launcher,D0_completed_episodes=34,
        D0_completion_sha256=completions,D0_output_files_verified=files,gpu_compute_processes=[],disk_free_bytes=free,
        expected_peak_gpu='One model process at a time; previous B peak allocated 13.46 GB on 16 GB GPU',
        sequence=['evaluate_reference.py','train_l.py --arm L --run'])


def worker(args):
    gate(args)
    started=time.monotonic()
    stages=[('B8000_extra_reference','evaluate_reference.py',[]),('L_training','train_l.py',['--arm','L','--run'])]
    try:
        for stage,script,extra in stages:
            gpu_empty()
            command=[str(PYTHON),'-B','-u',str(ROOT/script),*extra]
            save(ROOT/'sequence_status.json',dict(status='RUNNING',stage=stage,pid=os.getpid(),updated_utc=stamp(),command=command),replace=True)
            with (ROOT/f'console_{stage}.log').open('xb') as stream:
                result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
            if result.returncode:raise RuntimeError(f'{stage} exited {result.returncode}; outputs retained for review')
            if stage=='B8000_extra_reference':
                d=json.loads((ROOT/'reference_B8000_extra/completed.json').read_text())
                assert d['status']=='COMPLETE' and d['windows']==144 and d['overlap_old_panel_windows']==49
            else:
                d=json.loads((ROOT/'arms/L/training_summary.json').read_text())
                assert d['status']=='COMPLETE' and d['completed_updates']==8000 and d['frozen_vlm_unchanged']
                assert json.loads((ROOT/'arms/L/baseline_verification.json').read_text())['status']=='PASS'
                assert json.loads((ROOT/'arms/L/first_100_verification.json').read_text())['status']=='PASS'
        gpu_empty()
        save(ROOT/'sequence_status.json',dict(status='COMPLETE',pid=os.getpid(),updated_utc=stamp(),elapsed_s=time.monotonic()-started,
            completed_updates=8000,reference_complete=True,launcher_sha256=args.approved_launcher),replace=True)
    except BaseException as error:
        save(ROOT/'sequence_status.json',dict(status='FAILED',pid=os.getpid(),updated_utc=stamp(),error=f'{type(error).__name__}: {error}'),replace=True)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--check',action='store_true');mode.add_argument('--launch',action='store_true');mode.add_argument('--worker',action='store_true')
    p.add_argument('--approved-prepared',required=True);p.add_argument('--approved-d0',required=True);p.add_argument('--approved-launcher',required=True)
    args=p.parse_args()
    if args.worker:return worker(args)
    checked=gate(args)
    if args.check:print(json.dumps(checked,indent=2));return
    assert not (ROOT/'sequence_launch_receipt.json').exists(),'Existing sequence receipt; do not duplicate launch'
    assert not (ROOT/'arms/L/run_manifest.json').exists(),'Existing L run; explicit resume review required'
    assert not (ROOT/'reference_B8000_extra/completed.json').exists(),'Existing reference; explicit continuation review required'
    save(ROOT/'sequence_launch_gate.json',checked)
    command=[str(PYTHON),'-B','-u',str(Path(__file__).resolve()),'--worker','--approved-prepared',args.approved_prepared,'--approved-d0',args.approved_d0,'--approved-launcher',args.approved_launcher]
    with (ROOT/'sequence_console.log').open('xb') as stream:
        proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    receipt=dict(status='STARTED',pid=proc.pid,started_utc=stamp(),command=command,launcher_sha256=args.approved_launcher,
        prepared_sha256=args.approved_prepared,D0_protocol_sha256=args.approved_d0,gate_sha256=sha(ROOT/'sequence_launch_gate.json'))
    save(ROOT/'sequence_launch_receipt.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
