import bootstrap,json
import torch
from history_inputs import HistoryInputs
from recovery_common import ROOT,XR0,write
rows=[json.loads(l) for l in (ROOT/'schedule_H_4000.jsonl').read_text().splitlines()];i=next(i for i,r in enumerate(rows) if r['source']=='R1' and r['frame']==0);inputs=HistoryInputs(rows,'H0');report={}
for mode in ['H0','Hrepeat','Hhistory']:
 inputs.mode=mode;inputs.cache.clear();batch=inputs.get(i);state=batch['state'];assert state.shape==(1,1 if mode=='H0' else 4,32) and state.dtype==torch.float32 and torch.isfinite(state).all();assert batch['action'].shape==(1,30,32)
 if mode=='Hrepeat':assert torch.equal(state[:,0],state[:,1]) and torch.equal(state[:,0],state[:,2]) and torch.equal(state[:,0],state[:,3])
 if mode=='Hhistory':assert not torch.equal(state[:,0],state[:,-1])
 report[mode]={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in batch.items()}
write(ROOT/'batch_check.json',dict(status='PASS',anchor=rows[i]['window_id'],conditions=report));print(json.dumps(dict(status='PASS',shapes={m:report[m]['state'] for m in report})))
