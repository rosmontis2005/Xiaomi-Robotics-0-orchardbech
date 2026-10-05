"""Execute one explicitly selected Round2/3 S/J comparison, then stop."""
import argparse, json, os, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PYTHON=ROOT.parent/'.venv-orchard/bin/python'

def run(round_number,name,script,*args):
    command=[str(PYTHON),'-u','-B',str(ROOT/script),*map(str,args)]
    with (ROOT/f'round{round_number}_{name}.log').open('ab',buffering=0) as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    (ROOT/f'round{round_number}_{name}_receipt.json').write_text(json.dumps(dict(pid=child.pid,command=command,time=time.time()),indent=2))
    (ROOT/'pipeline_status.json').write_text(json.dumps(dict(status='RUNNING',round=round_number,stage=name,pid=os.getpid(),child_pid=child.pid,time=time.time(),maximum_rounds=3,fresh24_used=False),indent=2))
    code=child.wait()
    if code:raise RuntimeError(f'Round{round_number} {name} exited {code}; evidence retained')

def main(round_number):
    previous=json.loads((ROOT/'pipeline_status.json').read_text())
    assert previous['status']==f'ROUND{round_number-1}_COMPLETE',previous
    frozen=json.loads((ROOT/f'round{round_number}/frozen.json').read_text())
    assert frozen['status']=='FROZEN' and frozen['round']==round_number
    try:
        run(round_number,'S_train','train_integrated_next.py','--round',round_number,'--arm','S')
        run(round_number,'S8_eval','evaluate_next.py','--round',round_number,'--endpoint','S8','--checkpoint',ROOT/f'round{round_number}/S/training/checkpoints/step_8000_trainable.pt')
        run(round_number,'J_train','train_integrated_next.py','--round',round_number,'--arm','J')
        for endpoint,step in [('J8',8000),('J16',16000)]:
            run(round_number,f'{endpoint}_eval','evaluate_next.py','--round',round_number,'--endpoint',endpoint,'--checkpoint',ROOT/f'round{round_number}/J/training/checkpoints/step_{step:04d}_trainable.pt')
        run(round_number,'summary','summarize_round.py','--round',round_number)
        (ROOT/'pipeline_status.json').write_text(json.dumps(dict(status=f'ROUND{round_number}_COMPLETE',completed_rounds=round_number,maximum_rounds=3,fresh24_used=False,time=time.time()),indent=2))
    except BaseException as error:
        (ROOT/'pipeline_status.json').write_text(json.dumps(dict(status='FAILED',round=round_number,error=str(error),time=time.time()),indent=2))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--round',type=int,choices=[2,3],required=True)
    main(parser.parse_args().round)
