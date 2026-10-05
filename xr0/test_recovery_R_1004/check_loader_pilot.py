import bootstrap,json,copy
import torch
from recovery_common import ROOT,write
from recovery_dataset import RecoveryDataset
from mibot.data.datamodule.orchardbench_datamodule import OrchardBenchDataModule
from recovery_common import PROCESSOR
rows=[]
for c in json.loads((ROOT/'mechanism_gate.json').read_text())['checks']:rows.append(json.loads((ROOT/'recoveries'/c['candidate_id']/'decision.json').read_text()))
p=ROOT/'mechanism_manifest.jsonl'
p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
dataset=RecoveryDataset(p);dm=OrchardBenchDataModule(dict(processor_path=str(PROCESSOR)));checks=[]
for row in rows:
 t=dataset.episodes[row['annotation']];sample=dataset.from_trajectory(t,0);assert set(sample)=={'messages','action','action_mask','state'}
 batch=dm.collate_fn([sample]);assert batch['action'].shape==(1,30,32) and batch['action'].dtype==torch.float32 and torch.isfinite(batch['action']).all()
 changed=copy.deepcopy(t);changed['recovery']={'fruit_xyz':[999.,999.,999.]};changed['orchardbench']={};other=dm.collate_fn([dataset.from_trajectory(changed,0)])
 assert all(torch.equal(v,other[k]) for k,v in batch.items())
 checks.append(dict(candidate=row['candidate_id'],category=row['category'],tensor_shapes={k:list(v.shape) for k,v in batch.items()},privileged_metadata_invariant=True))
write(ROOT/'loader_pilot_check.json',dict(status='PASS',checks=checks,cuda_initialized=torch.cuda.is_initialized()));print('six trajectory loader and metadata-isolation checks PASS')
