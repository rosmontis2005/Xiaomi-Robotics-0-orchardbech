"""Resume after reviewed CPU-only GT provider-name compatibility repair."""
import bootstrap
import argparse,json,subprocess,sys,time
from pathlib import Path
import evaluate_grasp as e
p=argparse.ArgumentParser();p.add_argument('--approved-protocol',required=True);args=p.parse_args()
assert e.sha(e.PROTOCOL)==args.approved_protocol
e.runtime_checks()
assert not (e.ROOT/'resume_001_receipt.json').exists()
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
assert not gpu.stdout.strip(),gpu.stdout
cmd=[sys.executable,'-B','-u',str(e.ROOT/'evaluate_grasp.py'),'--queue']
with (e.ROOT/'queue_resume001_console.log').open('x') as out:
 process=subprocess.Popen(cmd,cwd=e.ROOT,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
e.save_new(e.ROOT/'resume_001_receipt.json',dict(pid=process.pid,command=cmd,protocol_sha256=args.approved_protocol,started_unix=time.time(),reason='CPU postprocessing provider-name alias only; first GT physics trace reused unchanged'))
print(json.dumps(dict(pid=process.pid,protocol_sha256=args.approved_protocol)))
