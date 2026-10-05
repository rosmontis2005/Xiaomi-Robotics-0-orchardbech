import bootstrap
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from recovery_common import ROOT,write
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('script');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
cmd=[str(ROOT.parent/'.venv-orchard/bin/python'),'-u','-B',str(ROOT/a.script),*a.args]
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
with (ROOT/f'{a.name}.log').open('ab',buffering=0) as out:
 proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
r=dict(pid=proc.pid,session=proc.pid,command=cmd,log=str(ROOT/f'{a.name}.log'),detached=True,time=time.time());write(ROOT/f'{a.name}_receipt.json',r);print(json.dumps(r))
