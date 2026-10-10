from common import *
from prepare import profile
from collections import defaultdict
import numpy as np

def run():
 assert not (ROOT/'frozen.json').exists()
 rng=np.random.default_rng(42);windows=lines(ROOT/'window_index.jsonl');schedule=[]
 for group,count in {'reset':3000,'early':9000,'grasp':12000,'pull':6000,'transport':15000,'drop':12000,'transition':3000}.items():
  pool=defaultdict(list)
  for w in windows:
   if w['split']=='train' and w['group']==group:pool[w['seed']].append(w)
  cycles=[]
  while len(cycles)<count:cycles.extend(rng.permutation(sorted(pool)).tolist())
  for j,seed in enumerate(cycles[:count]):
   eligible=pool[seed]
   if group=='grasp' and j%2==0:
    # Closing or windows with closing in future30, including contact approach.
    with np.load(ROOT/'arrays'/f'{seed}.npz') as a:
     contact=[w for w in eligible if (a['widths'][w['frame']:w['frame']+30]<.04).any()]
    if contact:eligible=contact
   if group=='drop' and j%2==0:
    with np.load(ROOT/'arrays'/f'{seed}.npz') as a:drop=int(np.flatnonzero(a['phases']==4)[0])
    near=[w for w in eligible if drop-15<=w['frame']<=drop+10]
    if near:eligible=near
   schedule.append(dict(eligible[int(rng.integers(len(eligible)))]))
 rng.shuffle(schedule)
 for i,r in enumerate(schedule):r['step']=i+1
 for name in ['schedule.jsonl','planned_exposure.json']:(ROOT/name).rename(ROOT/('initial_'+name))
 jsonl(ROOT/'schedule.jsonl',schedule);write(ROOT/'planned_exposure.json',profile(schedule))
 write(ROOT/'schedule_revision.json',dict(reason='GRASP first5 closing only 7.6%; reserve half group draws for windows reaching closing within full30. Reserve half drop draws within -15/+10 of release command onset.',group_ratios_unchanged=True,only_pretraining_revision=True,episode_cycles='balanced shuffled cycles within each mutually exclusive group',reset_updates=3000))
 # 256 per split, balanced phases plus reset16. contact subset half GRASP.
 panels=[]
 for split in ['train','validation']:
  for phase,num in [('reset',16),('REACH',36),('GRASP',51),('PULL',51),('TRANSPORT',51),('DROP',51)]:
   by=defaultdict(list)
   for w in windows:
    if w['split']==split and ((w['frame']==0) if phase=='reset' else (w['anchor_phase']==phase and w['frame']>0)):by[w['seed']].append(w)
   for j,seed in enumerate(rng.choice(sorted(by),num,replace=False)):
    eligible=by[seed]
    if phase=='GRASP' and j%2==0:
     contact=[w for w in eligible if w['closing_first5']>0]
     if contact:eligible=contact
    panels.append(dict(eligible[int(rng.integers(len(eligible)))],eval_seed=170000+len(panels),panel_stratum=phase))
 (ROOT/'panels.jsonl').rename(ROOT/'initial_panels.jsonl');jsonl(ROOT/'panels.jsonl',panels)
if __name__=='__main__':run()
