"""Explicit root-reviewed D0 launch gate; no automatic training launch."""
import bootstrap
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import evaluate_grasp as evaluation

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--approved-protocol',required=True);args=parser.parse_args()
 root=bootstrap.ROOT
 assert not (root/'launch_receipt.json').exists(),'Already launched; inspect existing queue status'
 assert evaluation.sha(evaluation.PROTOCOL)==args.approved_protocol,'Explicit reviewed hash mismatch'
 protocol=evaluation.runtime_checks()
 gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 assert not gpu.stdout.strip(),('GPU busy',gpu.stdout)
 gate=dict(status='PASS',checked_unix=time.time(),approved_protocol_sha256=args.approved_protocol,source_hashes_verified=True,annotations_verified=True,checkpoint_sha256=protocol['checkpoint']['sha256'],gpu_compute_processes=[],primary_development_episodes=32,historical_success_regression_episodes=2)
 evaluation.save_new(root/'launch_gate.json',gate)
 command=[sys.executable,'-B','-u',str(root/'evaluate_grasp.py'),'--queue']
 with (root/'queue_console.log').open('x') as output:
  process=subprocess.Popen(command,cwd=root,stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 receipt=dict(status='STARTED',pid=process.pid,command=command,protocol_sha256=args.approved_protocol,started_unix=time.time(),console=str(root/'queue_console.log'))
 evaluation.save_new(root/'launch_receipt.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
