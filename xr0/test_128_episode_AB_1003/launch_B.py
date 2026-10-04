#!/usr/bin/env python3
"""Explicit detached launch of B only, after ROOT reviews preparation and releases GPU.

This command does not launch A or queue any post-training evaluation.
"""
import bootstrap
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=bootstrap.ROOT
PYTHON=ROOT.parent/'.venv-orchard/bin/python'

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--launch',action='store_true',required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not PYTHON.is_file():raise FileNotFoundError(PYTHON)
    from train_ab import ensure_precision_ready
    ensure_precision_ready()
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['CUDA_VISIBLE_DEVICES']=''
    checked=subprocess.run([str(PYTHON),'-B',str(ROOT/'train_ab.py'),'--check-prepared'],cwd=ROOT,env=env,check=True,text=True,capture_output=True)
    print(checked.stdout,flush=True)
    out=ROOT/'arms/B';out.mkdir(parents=True,exist_ok=True)
    if (out/'run_manifest.json').exists() and not args.resume:raise FileExistsError('Existing B run; use explicit --resume')
    status=out/'status.json'
    if status.exists():
        previous=json.loads(status.read_text());pid=previous.get('pid')
        if pid:
            try:
                os.kill(pid,0)
            except ProcessLookupError:pass
            else:raise RuntimeError(f'Existing B PID {pid} may still be alive')
    command=[str(PYTHON),'-u','-B',str(ROOT/'train_ab.py'),'--arm','B','--run']
    if args.resume:command.append('--resume')
    env['CUDA_VISIBLE_DEVICES']='0'
    with (out/'console.log').open('ab',buffering=0) as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    receipt=dict(arm='B',pid=proc.pid,command=command,resume=args.resume,created_utc=datetime.now(timezone.utc).isoformat(),
                 console=str(out/'console.log'),status=str(status),detached_session=True,A_queued=False)
    destination=out/'launch_receipt.json';temporary=destination.with_name(destination.name+'.pending')
    temporary.write_text(json.dumps(receipt,indent=2)+'\n');os.replace(temporary,destination)
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
