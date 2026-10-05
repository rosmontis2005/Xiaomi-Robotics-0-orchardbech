"""Execute each fixed H training immediately followed by its fixed evaluation."""
import argparse,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--updates',type=int,choices=[4000,8000],required=True);a=p.parse_args();root=Path(__file__).resolve().parent;python=root.parent/'.venv-orchard/bin/python';decision=json.loads((root/'budget_decision.json').read_text());assert decision['H_updates']==a.updates
for condition in ['H0','Hrepeat','Hhistory']:
 cmds=[[str(python),'-u','-B',str(root/'setup_condition.py'),condition,'--updates',str(a.updates)], [str(python),'-u','-B',str(root/'train_H.py'),'--condition',condition,'--updates',str(a.updates),'--run'], [str(python),'-u','-B',str(root/'evaluate_followup.py'),'--condition',condition,'--checkpoint',str(root/condition/f'training/checkpoints/step_{a.updates:04d}_trainable.pt'),'--offline'], [str(python),'-u','-B',str(root/'summarize_cases.py'),str(root/condition/'evaluation')]]
 for i,cmd in enumerate(cmds):
  receipt=dict(command=cmd,time=time.time());print(json.dumps(receipt),flush=True)
  with (root/f'{condition}_stage{i}_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
  with (root/f'{condition}_stage{i}.log').open('xb',buffering=0) as f:proc=subprocess.Popen(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT);receipt['pid']=proc.pid;(root/f'{condition}_stage{i}_receipt.json').write_text(json.dumps(receipt,indent=2));code=proc.wait()
  if code:raise RuntimeError(f'{condition} stage{i} failed with exit {code}; retained log and partial artifacts')
 subprocess.run([str(python),'-u','-B',str(root/'collect_training_summary.py'),str(root/condition)],check=True)
 print(json.dumps(dict(event='CONDITION_COMPLETE',condition=condition,aggregate=json.loads((root/condition/'evaluation/aggregate.json').read_text()))),flush=True)
print('THREE_H_CONDITIONS_COMPLETE; optional H4 requires explicit evidence decision',flush=True)
