import json,hashlib,sys
from pathlib import Path
from collections import Counter
import numpy as np
root=Path(__file__).resolve().parent

def summarize(folder,prior=None):
 folder=Path(folder);rows=[json.loads(l) for l in (folder/'training/training_steps.jsonl').read_text().splitlines()];manifest=json.loads((folder/'training/run_manifest.json').read_text());status=json.loads((folder/'training/status.json').read_text());step=status['completed_updates'];checkpoint=folder/f'training/checkpoints/step_{step:04d}_trainable.pt';h=hashlib.sha256()
 with checkpoint.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 allrows=rows if prior is None else [json.loads(l) for l in Path(prior).read_text().splitlines()]+rows
 sources=Counter(x['source'] for x in allrows)
 def losses(data):
  return {s:dict(updates=sum(r['source']==s for r in data),mean=float(np.mean([r['flow_loss'] for r in data if r['source']==s])),first500_mean=float(np.mean([r['flow_loss'] for r in data[:500] if r['source']==s])),last500_mean=float(np.mean([r['flow_loss'] for r in data[-500:] if r['source']==s]))) for s in sources}
 result=dict(status=status,checkpoint=dict(path=str(checkpoint),sha256=h.hexdigest(),step=step),updates_in_this_run=len(rows),total_actual_updates=len(allrows),source_counts_total=dict(sources),source_counts_this_run=dict(Counter(r['source'] for r in rows)),loss_by_source_total=losses(allrows),loss_by_source_this_run=losses(rows),actual_step_sequence=[rows[0]['step'],rows[-1]['step']],all_real_updates=all(r['real_optimizer_update'] for r in allrows),constant_lr=sorted({r['lr'] for r in allrows}),manifest=manifest)
 (folder/'training/summary.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
 result=summarize(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None);print(json.dumps({k:result[k] for k in ['checkpoint','total_actual_updates','source_counts_total','constant_lr']}))
