"""Separate CPU preflight and root-authorized post-primary B train-best launch."""
import bootstrap
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import evaluate_ab as evaluation
ROOT=bootstrap.ROOT
EXPECTED_PROTOCOL='e1f4f9fc6368cb02cd39503a2d6b5713283c9363d55d375a818e9b041f55ae9a'

def inspect():
 import torch
 assert evaluation.sha(evaluation.PROTOCOL)==EXPECTED_PROTOCOL
 protocol,endpoints=evaluation.runtime_checks()
 records={arm:json.loads((bootstrap.AB/'arms'/arm/'best_checkpoint.json').read_text()) for arm in ['A','B']}
 assert records['A']['step']==8000 and records['B']['step']==4000
 jobs=[dict(j) for j in protocol['jobs']['B'] if j['scene']['cohort']=='dev']
 assert len(jobs)==8 and all(j['inference_seed']==42 for j in jobs)
 checks=[]
 for j in jobs:
  s=j['scene'];assert evaluation.sha(s['annotation'])==s['annotation_sha256']
  checks.append(dict(scene_seed=s['seed'],episode_id=s['episode_id'],annotation_sha256=s['annotation_sha256'],inference_seed=42))
 r=records['B'];path=Path(r['path']);assert path==bootstrap.AB/'arms/B/checkpoints/step_4000_trainable.pt'
 assert r['rule']['criterion']=='lexicographic, selected TRAIN panel only; VAL is report-only'
 assert evaluation.sha(path)==r['checkpoint_sha256']
 payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
 assert payload['step']==4000 and payload['format']=='orchard_m0_trainable_v1'
 assert payload['metadata']['provenance']['arm']=='B'
 assert payload['metadata']['provenance']['selection_sha256']==protocol['selection_sha256']
 assert Path(payload['base_checkpoint']).resolve()==evaluation.BASE.resolve()
 assert payload['base_sha256']==endpoints['B']['base_sha256']
 assert not torch.cuda.is_initialized()
 return dict(status='PASS',protocol_sha256=EXPECTED_PROTOCOL,arm='B',step=4000,checkpoint_path=str(path),checkpoint_sha256=r['checkpoint_sha256'],best_record_sha256=evaluation.sha(bootstrap.AB/'arms/B/best_checkpoint.json'),A_best_step8000_reused=True,primary_endpoint_unchanged=8000,cohort='dev',inference_seeds=[42],scenes=checks,cuda_initialized=False,simulator_imported=any(k=='treesim.vla_env' for k in sys.modules),source_sha256=protocol['source_sha256'],comparison_role='secondary_train_best',control_path='Frozen evaluate_ab.run -> unchanged M0 episode; full30 reach .01m/.08rad, dwell30, budget900',authorization='Root separately authorized after complete primary84 integrity and empty GPU; B train-only best4000 dev8 only; no heldout selection or replacement of primary endpoint')

def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--preflight',action='store_true');g.add_argument('--launch',action='store_true');args=p.parse_args()
 report=inspect();assert not report['simulator_imported']
 preflight=ROOT/'secondary_cpu_preflight.json'
 if args.preflight:
  evaluation.save_new(preflight,dict(report,checked_unix=time.time(),launcher_sha256=evaluation.sha(__file__)));print(json.dumps(report));return
 previous=json.loads(preflight.read_text());assert previous['launcher_sha256']==evaluation.sha(__file__)
 assert all(previous[k]==v for k,v in report.items())
 assert json.loads((ROOT/'queue_status.json').read_text())['status']=='COMPLETE'
 protocol=json.loads(evaluation.PROTOCOL.read_text());endpoints=json.loads((ROOT/'endpoint_checkpoints.json').read_text())
 counts={}
 for stage,n in [('GT',20),('A',32),('B',32)]:
  marker=json.loads((ROOT/f'stage_{stage}_complete.json').read_text());assert marker['status']=='COMPLETE' and marker['episodes']==n and marker['protocol_sha256']==EXPECTED_PROTOCOL
  digest=None if stage=='GT' else endpoints[stage]['sha256']
  for j in protocol['jobs'][stage]:assert evaluation.completion_for(j,digest) is not None
  counts[stage]=n
 primary_pid=json.loads((ROOT/'launch_receipt.json').read_text())['pid']
 proc=Path('/proc')/str(primary_pid)
 if proc.exists():assert (proc/'stat').read_text().rsplit(')',1)[1].strip().split()[0]=='Z'
 gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
 active=[s for s in gpu.stdout.splitlines() if s.strip()];assert not active
 gate=dict(report,checked_unix=time.time(),primary84_all_completed_raw_hashes_valid=True,primary_counts=counts,primary_pid_exited_or_zombie=True,gpu_compute_processes=active,preflight_sha256=evaluation.sha(preflight))
 evaluation.save_new(ROOT/'secondary_launch_gate.json',gate)
 cmd=[sys.executable,'-B','-u',str(ROOT/'evaluate_ab.py'),'--run','B','--secondary-train-best']
 with (ROOT/'console_B_trainbest.log').open('x') as out:
  process=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 receipt=dict(status='STARTED',pid=process.pid,command=cmd,started_unix=time.time(),secondary_episode_count=8,comparison_role='secondary_train_best',launch_gate_sha256=evaluation.sha(ROOT/'secondary_launch_gate.json'))
 evaluation.save_new(ROOT/'secondary_launch_receipt.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
