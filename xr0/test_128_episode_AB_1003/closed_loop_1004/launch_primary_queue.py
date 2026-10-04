"""Root-authorized launch gate. Run only after the training agent's GPU handoff."""
import bootstrap
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import evaluate_ab as evaluation

EXPECTED_PROTOCOL_SHA='e1f4f9fc6368cb02cd39503a2d6b5713283c9363d55d375a818e9b041f55ae9a'

def main():
 p=argparse.ArgumentParser();p.add_argument('--training-pid',type=int,required=True);args=p.parse_args()
 root=bootstrap.ROOT
 assert not (root/'launch_receipt.json').exists(),'Queue already launched; use existing completion/status records'
 assert evaluation.sha(evaluation.PROTOCOL)==EXPECTED_PROTOCOL_SHA
 protocol,endpoints=evaluation.runtime_checks()
 annotations=[]
 for job in protocol['jobs']['GT']:
  scene=job['scene'];actual=evaluation.sha(scene['annotation']);assert actual==scene['annotation_sha256']
  annotations.append(dict(seed=scene['seed'],episode_id=scene['episode_id'],annotation=scene['annotation'],sha256=actual))
 assert len(annotations)==20
 proc=Path('/proc')/str(args.training_pid)
 proc_exists=proc.exists();state=None;command=None
 if proc_exists:
  state=(proc/'stat').read_text().rsplit(')',1)[1].strip().split()[0]
  command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  assert state=='Z',('A training PID still live',args.training_pid,state,command)
 gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 active=[line for line in gpu.stdout.splitlines() if line.strip()]
 assert not active,('GPU compute processes remain',active)
 import torch
 assert not torch.cuda.is_initialized()
 gate=dict(status='PASS',checked_unix=time.time(),protocol_sha256=EXPECTED_PROTOCOL_SHA,
   a_training_pid=args.training_pid,a_training_pid_exists=proc_exists,a_training_process_state=state,a_training_command=command,
   both_arms_complete_step8000=True,endpoint_checkpoints=endpoints,annotations=annotations,
   gpu_compute_processes=active,nvidia_smi_stderr=gpu.stderr,cuda_initialized=False,
   launcher_sha256=evaluation.sha(Path(__file__)),authorization='Root reviewed protocol/source and authorized automatic primary queue after A completion, PID exit, empty GPU and annotation hash check; secondary excluded')
 evaluation.save_new(root/'launch_gate.json',gate)
 with (root/'queue_console.log').open('x') as output:
  command=[sys.executable,'-B','-u',str(root/'evaluate_ab.py'),'--queue']
  process=subprocess.Popen(command,cwd=root,stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 receipt=dict(status='STARTED',pid=process.pid,command=command,started_unix=time.time(),protocol_sha256=EXPECTED_PROTOCOL_SHA,launch_gate_sha256=evaluation.sha(root/'launch_gate.json'),primary_model_episodes=64,current_gt_episodes=20,secondary_launched=False,console=str(root/'queue_console.log'))
 evaluation.save_new(root/'launch_receipt.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
