"""Launch only after the root's training-to-evaluation GPU handoff."""
import bootstrap
import argparse,json,subprocess,sys,time
import evaluate_L as e
p=argparse.ArgumentParser();p.add_argument('--approved-protocol',required=True);args=p.parse_args()
assert e.sha(e.PROTOCOL)==args.approved_protocol
assert not (e.ROOT/'launch_receipt.json').exists()
e.runtime_checks()
r=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
assert not r.stdout.strip(),r.stdout
cmd=[sys.executable,'-B','-u',str(e.ROOT/'evaluate_L.py'),'--queue']
with (e.ROOT/'queue_console.log').open('x') as out:
 process=subprocess.Popen(cmd,cwd=e.ROOT,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
e.save_new(e.ROOT/'launch_receipt.json',dict(status='STARTED',pid=process.pid,command=cmd,protocol_sha256=args.approved_protocol,started_unix=time.time()))
print(json.dumps(dict(pid=process.pid,episodes=18)))
