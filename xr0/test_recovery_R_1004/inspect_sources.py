import json, hashlib
from pathlib import Path
X=Path('/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0'); R=X/'test_recovery_R_1004'; A=X/'test_128_episode_AB_1003'
s=json.loads((A/'selection.json').read_text()); m=json.loads((X/'test_small_set_fit_1003/selection.json').read_text())
print('AB keys',s.keys()); print('M0 keys',m.keys())
print('cohorts',dict(train=len(s['episodes_train']),dev=len(s['development_scenes']),heldout=len(s['heldout_scenes'])))
f=json.loads((X/'test_grasp_1004/data/fresh24_heldout.json').read_text());print('fresh keys',f.keys())
p=A/'arms/B/checkpoints/step_8000_trainable.pt'
import torch
b=torch.load(p,weights_only=True,map_location='cpu',mmap=True)
h=hashlib.sha256()
with p.open('rb') as file:
 for chunk in iter(lambda:file.read(8*1024*1024),b''):h.update(chunk)
print('checkpoint',str(p),b['step'],h.hexdigest(),b['base_sha256'],b['metadata'].keys()); print('base',b['base_checkpoint'])
assert b['step']==8000
summary=json.loads((A/'arms/B/training_summary.json').read_text());assert summary['checkpoint_records']['8000']['checkpoint_sha256']==h.hexdigest()
(R/'baseline_provenance.json').write_text(json.dumps(dict(checkpoint=str(p),step=b['step'],sha256=h.hexdigest(),base_checkpoint=b['base_checkpoint'],base_sha256=b['base_sha256'],training_summary=str(A/'arms/B/training_summary.json')),indent=2)+'\n')
