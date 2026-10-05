"""Wait only for the authorized continuation, then run its fixed evaluation."""
import json,subprocess,time,os
from pathlib import Path
root=Path(__file__).resolve().parent;train=root/'r03_extension/training/status.json'
while True:
 s=json.loads(train.read_text())
 if s['status']=='FAILED':raise RuntimeError(s)
 if s['status']=='COMPLETE':break
 time.sleep(5)
# Final status is written just before process teardown/lock release.
time.sleep(3)
cmd=[str(root.parent/'.venv-orchard/bin/python'),'-u','-B',str(root/'evaluate_followup.py'),'--condition','R03-8k','--checkpoint',str(root/'r03_extension/training/checkpoints/step_8000_trainable.pt'),'--offline']
(root/'r03_eval_command.json').write_text(json.dumps(dict(command=cmd,time=time.time()),indent=2));subprocess.run(cmd,check=True)
subprocess.run([str(root.parent/'.venv-orchard/bin/python'),'-u','-B',str(root/'summarize_cases.py'),str(root/'r03_extension/evaluation')],check=True)
