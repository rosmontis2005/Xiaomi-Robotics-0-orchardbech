import subprocess,json,sys,time
from pathlib import Path
root=Path('/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_history_H_1005')
name=sys.argv[1];script=Path(sys.argv[2]).resolve();cmd=[str(root.parent/'.venv-orchard/bin/python'),'-u','-B',str(script),*sys.argv[3:]]
with (root/f'{name}.log').open('ab',buffering=0) as log:p=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
receipt=dict(pid=p.pid,command=cmd,log=str(root/f'{name}.log'),time=time.time());(root/f'{name}_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
