"""Single data freeze then complete Round1; never autonomously open Round2/fresh24."""
import json,subprocess,sys,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PYTHON=ROOT.parent/'.venv-orchard/bin/python'
def run(name,script,*args):
    cmd=[str(PYTHON),'-u','-B',str(ROOT/script),*map(str,args)]
    with (ROOT/f'{name}.log').open('ab',buffering=0) as log:p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    (ROOT/f'{name}_receipt.json').write_text(json.dumps(dict(pid=p.pid,command=cmd,time=time.time()),indent=2))
    (ROOT/'pipeline_status.json').write_text(json.dumps(dict(status='RUNNING',stage=name,pid=os.getpid(),child_pid=p.pid,time=time.time()),indent=2))
    code=p.wait()
    if code:raise RuntimeError(f'{name} exited {code}; pipeline stops, evidence retained')
def main():
    try:
        if len(sys.argv)>1:
            pid=int(sys.argv[1])
            while True:
                try:os.kill(pid,0)
                except ProcessLookupError:break
                # Collection launched before this queue. Inspect status on each loop.
                stat=Path(f'/proc/{pid}/stat')
                if stat.exists() and stat.read_text().split()[2]=='Z':break
                time.sleep(3)
        else:run('collection','collect_live.py')
        run('freeze','freeze_data.py')
        run('S_train','train_integrated.py','--arm','S')
        run('S8_eval','evaluate.py','--endpoint','S8','--checkpoint',ROOT/'round1/S/training/checkpoints/step_8000_trainable.pt')
        run('J_train','train_integrated.py','--arm','J')
        run('J8_eval','evaluate.py','--endpoint','J8','--checkpoint',ROOT/'round1/J/training/checkpoints/step_8000_trainable.pt')
        run('J16_eval','evaluate.py','--endpoint','J16','--checkpoint',ROOT/'round1/J/training/checkpoints/step_16000_trainable.pt')
        (ROOT/'pipeline_status.json').write_text(json.dumps(dict(status='ROUND1_COMPLETE',completed_rounds=1,maximum_rounds=3,fresh24_used=False,time=time.time()),indent=2))
    except BaseException as e:
        (ROOT/'pipeline_status.json').write_text(json.dumps(dict(status='FAILED',error=str(e),time=time.time()),indent=2));raise
if __name__=='__main__':main()
