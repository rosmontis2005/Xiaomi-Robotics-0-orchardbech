from common import *
import numpy as np
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ProcessPoolExecutor
from collections import Counter,defaultdict
from treesim.orchard_command import encode_window,CONTRACT
from treesim.orchard_action import state_vector

def episode(row):
 p=Path(row['annotation']);raw=p.read_bytes();t=json.loads(raw);seed=row['seed'];starts=np.arange(0,t['num_commands']-29,5)
 assert len(starts)==row['windows']
 actions=np.stack([encode_window(t,int(f))[:,:7] for f in starts])
 states=np.stack([state_vector(o['tcp_pos_world'],R.from_quat(o['tcp_quat_world']).as_matrix(),o['gripper_width'],o['joint_pos'][:7]) for o in (t['observations'][f] for f in starts)])
 phases=np.array([PHASES.index(c['phase']) for c in t['commands']],np.int8)
 widths=np.array([c['width'] for c in t['commands']])
 changes=np.flatnonzero(phases[1:]!=phases[:-1])+1
 drop=np.flatnonzero(phases==4)[0];grasp=np.flatnonzero(phases==1)[0]
 pools=[]
 for f in starts:
  # Mutually exclusive: reset; early reach; closing/hold; detach; release;
  # selected upstream boundary windows; remaining transport/reach.
  phase=int(phases[f]);transition=any(f==b-5 for b in changes if phases[b]>=2)
  if f==0:g='reset'
  elif transition:g='transition'
  elif phase==0:g='early'
  elif phase==1:g='grasp'
  elif phase==2:g='pull'
  elif phase==4 or f>=drop-30:g='drop'
  else:g='transport'
  pools.append(g)
 # PULL may be <5 steps or immediately detach; reserve closest true detach
 # command window only if no pull pool, without duplicating classification.
 if 'pull' not in pools:
  candidates=[i for i,f in enumerate(starts) if np.any(phases[f:f+5]==2)]
  if candidates:pools[candidates[0]]='pull'
 cache=ROOT/'arrays'/f'{seed}.npz';cache.parent.mkdir(exist_ok=True)
 np.savez(cache,actions=actions,state=states,phases=phases,starts=starts,widths=widths,groups=np.array(pools))
 initial=json.loads((p.parent/'initial.json').read_text());tr=initial['truth'];o=t['observations'][0];br=R.from_quat(tr['base_pose'][3:]);delta=np.array(tr['fruit_position'])-o['tcp_pos_world']
 geom=dict(seed=seed,accepted=True,target_id=tr['target_id'],base_pose=tr['base_pose'],fruit_world=tr['fruit_position'],target_base=br.inv().apply(np.array(tr['fruit_position'])-tr['base_pose'][:3]).tolist(),tcp_world=o['tcp_pos_world'],tcp_quat=o['tcp_quat_world'],tcp_target_distance=float(np.linalg.norm(delta)),tcp_target_direction_base=br.inv().apply(delta/np.linalg.norm(delta)).tolist(),stance=initial['stance'])
 meta=dict(**row,episode=f'seed_{seed}',annotation_sha256=hashlib.sha256(raw).hexdigest(),array=str(cache),array_sha256=sha(cache),video_sources={k:str(p.parent/v) for k,v in t['rgb'].items()})
 sums=actions.astype('float64').sum(0);sq=(actions.astype('float64')**2).sum(0)
 windows=[dict(seed=seed,episode=f'seed_{seed}',frame=int(f),index=i,group=pools[i],anchor_phase=PHASES[phases[f]],first5_counts=np.bincount(phases[f:f+5],minlength=5).tolist(),full30_counts=np.bincount(phases[f:f+30],minlength=5).tolist(),closing_first5=int(np.sum(widths[f:f+5]<.04)),opening_first5=int(np.sum(widths[f:f+5]>.075))) for i,f in enumerate(starts)]
 return meta,geom,windows,sums,sq

def profile(rows):
 ec=Counter(r['episode'] for r in rows);wc=Counter((r['episode'],r['frame']) for r in rows)
 return dict(presentations=len(rows),unique_episodes=len(ec),unique_windows=len(wc),episode_counts=dict(ec),window_repeat_histogram=dict(Counter(wc.values())),repeated_windows=sum(n>1 for n in wc.values()),groups={g:dict(anchors=dict(Counter(r['anchor_phase'] for r in rr)),presentations=len(rr),first5=np.sum([r['first5_counts'] for r in rr],axis=0).tolist(),full30=np.sum([r['full30_counts'] for r in rr],axis=0).tolist(),closing_first5=sum(r['closing_first5'] for r in rr),opening_first5=sum(r['opening_first5'] for r in rr)) for g in sorted({r['group'] for r in rows}) if (rr:=[r for r in rows if r['group']==g])})

def main():
 assert not (ROOT/'train_manifest.jsonl').exists(),'Frozen preparation exists'
 raw=lines(DATA/'accepted_manifest.jsonl');assert len(raw)==2000 and sum(r['windows'] for r in raw)==483284
 seeds=sorted(r['seed'] for r in raw);rng=np.random.default_rng(42);val=set(rng.permutation(seeds)[:200].tolist())
 train=[];validation=[];geometry=[];windows=[];sums=np.zeros((30,7));squares=sums.copy();n=0
 with ProcessPoolExecutor(max_workers=4) as pool:
  for i,(meta,g,ws,s,q) in enumerate(pool.map(episode,raw)):
   meta['split']='validation' if meta['seed'] in val else 'train';(validation if meta['seed'] in val else train).append(meta)
   geometry.append(g);windows.extend([dict(**w,split=meta['split']) for w in ws])
   if meta['split']=='train':sums+=s;squares+=q;n+=meta['windows']
   if i%100==0:print('PREPARE',i,flush=True)
 jsonl(ROOT/'train_manifest.jsonl',train);jsonl(ROOT/'validation_manifest.jsonl',validation)
 mean=np.zeros((30,32));std=np.ones((30,32));mean[:,:7]=sums/n;std[:,:7]=np.maximum(np.sqrt(np.maximum(squares/n-mean[:,:7]**2,0)),1e-4)
 write(ROOT/'action_stats.json',dict(contract=CONTRACT,source_split='train',train_manifest_sha256=sha(ROOT/'train_manifest.jsonl'),windows=n,mean=mean.tolist(),std=std.tolist(),epsilon=1e-6,active_dims=list(range(7))))
 jsonl(ROOT/'window_index.jsonl',windows)
 # Balanced episode cycles per group. Fixed exact quotas shuffled once.
 quotas={'reset':3000,'early':9000,'grasp':12000,'pull':6000,'transport':15000,'drop':12000,'transition':3000}
 schedule=[]
 for group,count in quotas.items():
  pool=defaultdict(list)
  for w in windows:
   if w['split']=='train' and w['group']==group:pool[w['seed']].append(w)
  assert pool,group
  cycle=[]
  while len(cycle)<count:cycle.extend(rng.permutation(sorted(pool)).tolist())
  for seed in cycle[:count]:schedule.append(dict(pool[seed][int(rng.integers(len(pool[seed]))) ]))
 rng.shuffle(schedule)
 for i,row in enumerate(schedule):row['step']=i+1
 jsonl(ROOT/'schedule.jsonl',schedule);write(ROOT/'planned_exposure.json',profile(schedule))
 panels=[]
 for split in ['train','validation']:
  for phase,num in zip(PHASES,[52,51,51,51,51]):
   pool=[w for w in windows if w['split']==split and w['anchor_phase']==phase];assert len(pool)>=num
   # Episode stratification first; panel is independent of training schedule.
   by=defaultdict(list)
   for w in pool:by[w['seed']].append(w)
   chosen=rng.choice(sorted(by),size=num,replace=len(by)<num)
   for seed in chosen:panels.append(dict(by[seed][int(rng.integers(len(by[seed])))],eval_seed=170000+len(panels)))
 jsonl(ROOT/'panels.jsonl',panels)
 # rejected but valid resets: only initial record and first saved observation.
 accepted=set(seeds)
 for p in sorted(DATA.glob('seed_*/initial.json')):
  seed=int(p.parent.name.split('_')[1])
  if seed in accepted:continue
  t=json.loads(p.read_text())['truth'];br=R.from_quat(t['base_pose'][3:]);first=json.loads((p.parent/'steps.jsonl').open().readline())['observation_before'];d=np.array(t['fruit_position'])-first['tcp_pos_world']
  geometry.append(dict(seed=seed,accepted=False,target_id=t['target_id'],target_base=br.inv().apply(np.array(t['fruit_position'])-t['base_pose'][:3]).tolist(),tcp_target_distance=float(np.linalg.norm(d)),tcp_target_direction_base=br.inv().apply(d/np.linalg.norm(d)).tolist()))
 write(ROOT/'geometry.json',geometry)
 # 150 geometry-stratified samples; sort by z then x with distance tie breaker.
 ordered=sorted([g for g in geometry if g['accepted']],key=lambda g:(g['target_base'][2],g['target_base'][0],g['tcp_target_distance']))
 write(ROOT/'visibility_selection.json',[ordered[i]['seed'] for i in np.linspace(0,len(ordered)-1,150,dtype=int)])
 print('PREPARED',len(train),len(validation),n,flush=True)
if __name__=='__main__':main()
