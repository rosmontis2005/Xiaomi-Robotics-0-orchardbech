import bootstrap,json,subprocess,sys
from recovery_common import ROOT,write
assert json.loads((ROOT/'mechanism_gate.json').read_text())['status']=='PASS'
for script,args in [('mine_student.py',['--start','4','--count','10']),('collect_recovery.py',['--per-category','0'])]:
 write(ROOT/'expansion_status.json',dict(status='RUNNING',stage=script,pid=__import__('os').getpid()))
 subprocess.run([sys.executable,'-u','-B',str(ROOT/script),*args],cwd=ROOT,check=True)
write(ROOT/'expansion_status.json',dict(status='COMPLETE',pid=__import__('os').getpid(),training_queued=False))
