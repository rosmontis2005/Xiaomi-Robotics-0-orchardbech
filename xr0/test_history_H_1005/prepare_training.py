"""Reuse R03 episode/group balanced sampling; labels and pools unchanged."""
import bootstrap
import json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from recovery_common import XR0,ROOT,AB,sha,write
R03=XR0/'test_recovery_R_1004/evaluation_followup/version_03'
COUNTS=dict(original_B=2800,R1=500,R2=200,R3=200,R_continuation_teacher=300)
QUOTAS=dict(reset=560,first1_4=280,REACH=420,GRASP=700,PULL=420,TRANSPORT=280,DROP=140)
BRIDGE=dict(transport_to_drop=150,DROP_transition=100,DONE_settle=50)

def draw_episode_balanced(rng,options,quota,reserve_frame0=False):
 groups=defaultdict(list)
 for d,frames in options:
  for f in frames:groups[d['source_episode_id']].append((d,f))
 sources=sorted(groups);base,extra=divmod(quota,len(sources));draws=np.full(len(sources),base,dtype=int);draws[rng.permutation(len(sources))[:extra]]+=1;chosen=[]
 for source,n in zip(sources,draws):
  pool=groups[source];picked=[]
  if reserve_frame0:picked=[(d,0) for d in {d['candidate_id']:d for d,f in pool if f==0}.values()]
  assert n>=len(picked)
  chosen+=picked+[pool[int(i)] for i in rng.choice(len(pool),size=int(n)-len(picked),replace=True)]
 return chosen

def continuation(seed=1005):
 rng=np.random.default_rng(seed);selection=json.loads((AB/'selection.json').read_text());accepted=[json.loads(l) for l in (R03/'recovery_manifest.jsonl').read_text().splitlines()];rows=[]
 for group,quota in QUOTAS.items():
  choices=[(ep,ep['group_frames'][group]) for ep in selection['episodes_train'] if ep['group_frames'][group]];base,extra=divmod(quota,len(choices));draws=np.full(len(choices),base,dtype=int);draws[rng.permutation(len(choices))[:extra]]+=1
  for (ep,frames),n in zip(choices,draws):
   for f in rng.choice(frames,size=int(n),replace=True):rows.append(dict(source='original_B',group=group,episode_id=ep['episode_id'],source_episode_id=ep['episode_id'],seed=ep['seed'],split='train',frame=int(f),annotation=ep['annotation'],annotation_sha256=ep['annotation_sha256'],window_id=f"original_B/{ep['episode_id']}/frame{int(f):04d}"))
 rows=[rows[int(i)] for i in rng.permutation(len(rows))]
 def record(d,f,source,group):
  return dict(source=source,group=group,episode_id=d['candidate_id'],source_episode_id=d['source_episode_id'],seed=json.loads(Path(d['annotation']).read_text())['seed'],split='train',frame=int(f),annotation=d['annotation'],annotation_sha256=d['annotation_sha256'],window_id=f"{d['category']}/{d['candidate_id']}/frame{int(f):04d}",observation_start_role='student_takeover_neighbourhood' if source!='R_continuation_teacher' else 'teacher_continuation',recovery_origin_category=d['category'])
 for cat in ['R1','R2','R3']:
  for d,f in draw_episode_balanced(rng,[(d,d['takeover_anchors']) for d in accepted if d['category']==cat],COUNTS[cat],True):rows.append(record(d,f,cat,cat))
 for group,quota in BRIDGE.items():
  for d,f in draw_episode_balanced(rng,[(d,d['teacher_continuation_anchors_by_group'][group]) for d in accepted],quota):rows.append(record(d,f,'R_continuation_teacher',group))
 rows=[rows[int(i)] for i in rng.permutation(len(rows))]
 for i,row in enumerate(rows,4001):row['step']=i
 assert Counter(r['source'] for r in rows)==COUNTS
 return rows

if __name__=='__main__':
 rows=continuation();out=ROOT/'r03_extension/schedule_R.jsonl'
 with out.open('x') as f:
  for row in rows:f.write(json.dumps(row)+'\n')
 original=[json.loads(l) for l in (R03/'schedule_R.jsonl').read_text().splitlines()]
 with (ROOT/'schedule_H_8000.jsonl').open('x') as f:
  for row in original+rows:f.write(json.dumps(row)+'\n')
 (ROOT/'schedule_H_4000.jsonl').write_text((R03/'schedule_R.jsonl').read_text())
 write(ROOT/'schedule_manifest.json',dict(continuation_seed=1005,continuation_counts=COUNTS,continuation_sha256=sha(out),H4000_sha256=sha(ROOT/'schedule_H_4000.jsonl'),H8000_sha256=sha(ROOT/'schedule_H_8000.jsonl'),same_R03_pools=True,no_new_labels=True))
 print(json.dumps(dict(rows=len(rows),sources=COUNTS,sha256=sha(out))))
