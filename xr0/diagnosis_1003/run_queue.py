import bootstrap
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path
root=bootstrap.ROOT
jobs=[
 ('gt_time_first',['rollout_probe.py','--provider','gt','--modes','time-indexed','--seeds','2010027']),
 ('gt_remaining',['rollout_probe.py','--provider','gt','--modes','reach-conditioned','time-indexed','--seeds','2010042','2010067']),
 ('offline_inference',['offline_probe.py','--infer','--checkpoints','pretrained','step_10000']),
 ('model_rollouts',['rollout_probe.py','--provider','step_10000']),
]
with (root/'job_status.jsonl').open('x') as status:
 for name,args in jobs:
  if name=='model_rollouts':
   controls=[root/'rollouts'/f'gt_reach-conditioned_{seed}'/'summary.json' for seed in (2010027,2010042,2010067)]
   if not all(p.exists() and json.loads(p.read_text()).get('success') is True for p in controls):
    row=dict(job=name,status='skipped',reason='GT reach positive control did not pass 3/3')
    status.write(json.dumps(row)+'\n');status.flush();continue
  command=[sys.executable,'-B','-u',*args]
  start=time.monotonic()
  print('START '+name,flush=True)
  with (root/f'{name}_console.log').open('x') as log:
   result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
  row=dict(job=name,command=command,exit_code=result.returncode,elapsed_s=time.monotonic()-start,
           completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  status.write(json.dumps(row)+'\n');status.flush()
  print(json.dumps(row),flush=True)
