"""Sequentially evaluate the original and selected M0 policy after completed training."""
import bootstrap
import json
import subprocess
import sys
import time
from pathlib import Path
R=Path(__file__).resolve().parent
summary=json.loads((R/'training_summary.json').read_text())
assert summary['status']=='COMPLETE' and summary['completed_updates']==2000
jobs=[('original_10k',['--provider','original_10k','--video','first']),('best',['--provider','best','--checkpoint',str(R/'checkpoints/best_trainable.pt'),'--video','first'])]
with (R/'model_check_status.jsonl').open('x') as status:
    for name,args in jobs:
        started=time.time();row={'provider':name,'event':'started','time':started};status.write(json.dumps(row)+'\n');status.flush();print(json.dumps(row),flush=True)
        with (R/f'{name}_closed_loop_console.log').open('x') as log:
            p=subprocess.run([sys.executable,'-B','-u',str(R/'evaluate_reach.py'),*args],cwd=R.parent,stdout=log,stderr=subprocess.STDOUT)
        row={'provider':name,'event':'finished','returncode':p.returncode,'elapsed_s':time.time()-started};status.write(json.dumps(row)+'\n');status.flush();print(json.dumps(row),flush=True)
        if p.returncode: raise SystemExit(p.returncode)
    for script in ['analyze_reach.py','analyze_results.py']:
        p=subprocess.run([sys.executable,'-B',str(R/script)],cwd=R.parent)
        if p.returncode:raise SystemExit(p.returncode)
print('MODEL CHECK QUEUE COMPLETE',flush=True)
