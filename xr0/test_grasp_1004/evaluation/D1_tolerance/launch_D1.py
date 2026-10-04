"""Root-reviewed fixed D1 pilot, after all L18 evaluation work."""
import bootstrap
import argparse,json,subprocess,sys,time
import evaluate_D1 as e
p=argparse.ArgumentParser();p.add_argument('--approved-protocol',required=True);args=p.parse_args()
assert e.sha(e.PROTOCOL)==args.approved_protocol
assert not (e.ROOT/'launch_receipt.json').exists()
e.runtime_checks()
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
assert not gpu.stdout.strip(),gpu.stdout
cmd=[sys.executable,'-B','-u',str(e.ROOT/'evaluate_D1.py'),'--queue']
with (e.ROOT/'queue_console.log').open('x') as out:
 process=subprocess.Popen(cmd,cwd=e.ROOT,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
e.save_new(e.ROOT/'launch_receipt.json',dict(status='STARTED',pid=process.pid,command=cmd,protocol_sha256=args.approved_protocol,started_unix=time.time()))
print(json.dumps(dict(pid=process.pid,episodes=8)))
