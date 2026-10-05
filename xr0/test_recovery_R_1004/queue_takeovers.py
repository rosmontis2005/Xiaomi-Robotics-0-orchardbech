import bootstrap
import fcntl,os,json,sys,time
from pathlib import Path
root=bootstrap.ROOT
with (root/'.gpu_training.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 status=json.loads((root/'evaluation_followup/version_01/status.json').read_text())
 assert status['status']=='COMPLETE',status
os.execv(sys.executable,[sys.executable,'-u','-B',str(root/'evaluate_takeovers.py')])
