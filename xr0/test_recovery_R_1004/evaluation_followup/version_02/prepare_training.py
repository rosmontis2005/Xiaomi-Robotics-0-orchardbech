import bootstrap
import json,copy
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from recovery_common import *
from recovery_dataset import ScheduledInputs
from treesim.orchard_action import encode_window,decode_targets,EPS
QUOTAS=dict(reset=600,first1_4=300,REACH=450,GRASP=750,PULL=450,TRANSPORT=300,DROP=150)
COUNTS=dict(original_B=3000,R1=500,R2=250,R3=250)

def prepare():
 protection_check();decisions=[json.loads(p.read_text()) for p in sorted((Path('/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_recovery_R_1004')/'recoveries').glob('*/decision.json'))];accepted=[d for d in decisions if d['accepted']]
 categories=Counter(r['category'] for r in accepted);assert dict(categories)==PROTOCOL['pilot_targets'],('Pilot target not reached; explicitly document any small adjustment before freezing',dict(categories))
 unique_starts=[]
 for d in accepted:
  t=json.loads(Path(d['annotation']).read_text());r=t['recovery'];assert r['source_episode_id'] in {e['episode_id'] for e in PROTOCOL['source_episodes']}
  assert r['source_episode_id'] not in PROTOCOL['protected_episode_ids'] and r['source_scene_seed'] not in PROTOCOL['protected_scene_seeds']
  assert r['student_checkpoint_sha256']==PROTOCOL['baseline']['sha256'] and r['oracle_replay_result']['status']=='PASS'
  assert r['initialization_audit']['physics_unchanged'] and r['teacher_final_result']=='strict_success'
  start=(r['source_episode_id'],r['student_inference_seed'],r['student_control_step']);assert start not in unique_starts;unique_starts.append(start)
  assert 0 in d['legal_anchors'] and d['legal_anchors']==list(range(min(29,t['num_frames']-30)+1))
  if d['category']=='R1':
   teacher_steps=json.loads((Path(d['annotation']).parent/'teacher_steps.json').read_text());takeover=t['recovery']['takeover'];fruit=takeover['fruit_id']
   actual={0:takeover};actual.update({s['frame']//2:s['truth'] for s in teacher_steps})
   d['legal_anchors']=[frame for frame in d['legal_anchors'] if not actual[frame]['held'] and not actual[frame]['detached'][fruit]]
   assert 0 in d['legal_anchors'] and d['legal_anchors'],d['candidate_id']
 # Manifest includes accepted only, failures remain in candidate_decisions and directories.
 manifest=ROOT/'recovery_manifest.jsonl'
 with manifest.open('x') as f:
  for d in accepted:f.write(json.dumps(d)+'\n')
 selection=json.loads((AB/'selection.json').read_text());eps=selection['episodes_train'];rng=np.random.default_rng(42);B=[]
 # Identical B group allocation/frame draw/final shuffle logic; quarter quotas.
 for group,quota in QUOTAS.items():
  choices=[(ep,ep['group_frames'][group]) for ep in eps if ep['group_frames'][group]];count=len(choices);base,extra=divmod(quota,count);draws=np.full(count,base,dtype=int);draws[rng.permutation(count)[:extra]]+=1
  for (ep,frames),number in zip(choices,draws):
   B.extend(dict(source='original_B',episode_id=ep['episode_id'],source_episode_id=ep['episode_id'],seed=ep['seed'],split='train',frame=int(frame),group=group,annotation=ep['annotation'],annotation_sha256=ep['annotation_sha256'],window_id=f"original_B/{ep['episode_id']}/frame{int(frame):04d}") for frame in rng.choice(frames,size=int(number),replace=True))
 B=[B[int(i)] for i in rng.permutation(len(B))];assert Counter(r['group'] for r in B)==QUOTAS
 recovery=[];allocation={}
 for category in ['R1','R2','R3']:
  groups=defaultdict(list)
  for d in accepted:
   if d['category']==category:groups[d['source_episode_id']].append(d)
  sources=sorted(groups);base,extra=divmod(COUNTS[category],len(sources));draws=np.full(len(sources),base,dtype=int);draws[rng.permutation(len(sources))[:extra]]+=1;allocation[category]=dict(zip(sources,map(int,draws)))
  for source,n in zip(sources,draws):
   choices=[(d,f) for d in groups[source] for f in d['legal_anchors']]
   # Reserve frame0 of every accepted trajectory, then uniform draws over all
   # its episode's legal recovery anchors, with replacement.
   chosen=[(d,0) for d in groups[source]];assert n>=len(chosen)
   chosen += [choices[int(i)] for i in rng.choice(len(choices),size=int(n)-len(chosen),replace=True)]
   for d,frame in chosen:recovery.append(dict(source=category,group=category,episode_id=d['candidate_id'],source_episode_id=source,seed=next(e['seed'] for e in PROTOCOL['source_episodes'] if e['episode_id']==source),split='train',frame=frame,annotation=d['annotation'],annotation_sha256=d['annotation_sha256'],window_id=f"{category}/{d['candidate_id']}/frame{frame:04d}"))
 rows=B+recovery;rows=[rows[int(i)] for i in rng.permutation(len(rows))]
 for step,row in enumerate(rows,1):row['step']=step
 assert len(rows)==4000 and Counter(r['source'] for r in rows)==COUNTS
 schedule=ROOT/'schedule_R.jsonl'
 with schedule.open('x') as f:
  for r in rows:f.write(json.dumps(r)+'\n')
 inputs=ScheduledInputs(rows);tensor_checks={}
 for source in COUNTS:
  cursor=next(i for i,r in enumerate(rows) if r['source']==source);sample=inputs.sample(rows[cursor]);assert set(sample)=={'messages','action','action_mask','state'};batch=inputs.get(cursor)
  assert tuple(sample['action'].shape)==(30,32) and tuple(sample['state'].shape)==(1,32) and tuple(sample['action_mask'].shape)==(30,32)
  assert torch.isfinite(sample['action']).all() and torch.all(sample['action'][:,7:]==0)
  tensor_checks[source]={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in batch.items()}
 assert all(v==tensor_checks['original_B'] for v in tensor_checks.values())
 # Metadata mutation cannot change any collated policy tensor.
 for d in accepted:
  t=inputs.recovery.episodes[d['annotation']];modified=copy.deepcopy(t);modified['recovery']={'debug_usage':'poisoned metadata','fruit_xyz':[999,999,999]};modified['orchardbench']={'phase_by_frame':['poison']}
  a=inputs.dm.collate_fn([inputs.recovery.from_trajectory(t,0)]);b=inputs.dm.collate_fn([inputs.recovery.from_trajectory(modified,0)])
  assert set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a),'Privileged metadata entered batch'
 stats=json.loads(STATS.read_text());mean=np.asarray(stats['mean']);std=np.asarray(stats['std']);normalized=[];future_available=defaultdict(Counter);future_schedule=defaultdict(Counter);anchor_schedule=defaultdict(Counter);phase_cache={};outliers=[]
 for d in accepted:
  t=inputs.recovery.episodes[d['annotation']];n=t['num_frames'];p=t['proprios'];a=t['actions'];phase_cache[d['annotation']]=t['orchardbench']['phase_by_frame']
  for k in a:
   expected=np.asarray(p[k][1:]+p[k][-1:]);assert np.array_equal(a[k],expected),'Incorrect next measured-state indexing'
  assert np.allclose(np.diff(t['orchardbench']['timestamps']),1/30,atol=1e-9,rtol=0)
  for frame in d['legal_anchors']:
   action=encode_window(t,frame);xyz,rot,width=decode_targets(action,p['ee_pos'][frame],p['ee_rotm'][frame]);sl=slice(frame,frame+30)
   assert np.allclose(xyz,a['ee_pos'][sl],atol=2e-6,rtol=0) and np.allclose(rot,np.asarray(a['ee_rotm'][sl]).reshape(-1,3,3),atol=2e-6,rtol=0) and np.allclose(width,np.asarray(a['gripper_pos'][sl]).ravel(),atol=2e-6,rtol=0)
   z=(action-mean)/(std+EPS);normalized.append(z[:,:7]);maximum=float(np.abs(z[:,:7]).max())
   if maximum>5:outliers.append(dict(trajectory=d['candidate_id'],frame=frame,max_abs_normalized=maximum))
   phases=t['orchardbench']['phase_by_frame'];future_available[d['category']].update(phases[min(i,n-1)] for i in range(frame+1,frame+31))
 for row in rows:
  if row['source']=='original_B':
   if row['annotation'] not in phase_cache:
    t=json.loads(Path(row['annotation']).read_text());times=t['orchardbench']['timestamps'];trace=t['orchardbench']['fixed_base_expert']['state_trace'];event_times=np.array([v['frame']/60 for v in trace]);idx=np.searchsorted(event_times,np.asarray(times)+1e-7,side='right')-1;phase_cache[row['annotation']]=[trace[max(0,int(i))]['state'] for i in idx]
  phases=phase_cache[row['annotation']];anchor_schedule[row['source']].update([phases[row['frame']]]);future_schedule[row['source']].update(phases[min(i,len(phases)-1)] for i in range(row['frame']+1,row['frame']+31))
 exposure=Counter(r['window_id'] for r in rows if r['source']!='original_B');z=np.concatenate([v.reshape(-1,7) for v in normalized],axis=0)
 mined=[json.loads(p.read_text()) for p in (Path('/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_recovery_R_1004')/'student_sources').glob('*/complete.json')];selected=[json.loads(p.read_text()) for p in (Path('/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_recovery_R_1004')/'student_sources').glob('*/selected.json')]
 report=dict(status='PASS',candidate_counts=dict(Counter(d['category'] for d in decisions)),accepted_counts=dict(categories),rejected_counts=dict(Counter(d['category'] for d in decisions if not d['accepted'])),mined_selected_counts=dict(Counter(c['category'] for values in selected for c in values)),selected_but_not_attempted=[dict(candidate_id=c['candidate_id'],category=c['category'],source_episode_id=c['source_episode_id'],reason='category quota met before teacher attempt') for values in selected for c in values if c['candidate_id'] not in {d['candidate_id'] for d in decisions}],eligible_boundary_counts={c:sum(m['eligible_counts'][c] for m in mined) for c in ['R1','R2','R3']},unique_source_episodes=len({d['source_episode_id'] for d in accepted}),recovery_available_unique_windows=sum(len(d['legal_anchors']) for d in accepted),recovery_schedule_unique_windows=len(exposure),repeat_draws=1000-len(exposure),schedule_counts=dict(Counter(r['source'] for r in rows)),original_B_groups=dict(Counter(r['group'] for r in B)),schedule_sha256=sha(schedule),manifest_sha256=sha(manifest),schedule_seed=42,actual_anchor_phase_counts_schedule={k:dict(v) for k,v in anchor_schedule.items()},future_target_phase_counts_available={k:dict(v) for k,v in future_available.items()},future_target_phase_counts_schedule={k:dict(v) for k,v in future_schedule.items()},category_source_exposure=allocation,trajectory_exposure=dict(Counter(r['episode_id'] for r in recovery)),window_exposure=dict(exposure),source_exposure_all=dict(Counter(r['source_episode_id'] for r in rows)),tensor_contract=tensor_checks,privileged_metadata_mutation_invariance='PASS',next_state_indexing='PASS',roundtrip='PASS',teacher_and_oracle_all_accepted='PASS',normalization=dict(path=str(STATS),sha256=sha(STATS),recomputed=False,labels_clipped=False,max_abs_z_by_active_dim=np.abs(z).max(axis=0),abs_z_p99_by_active_dim=np.quantile(np.abs(z),.99,axis=0),outliers_over_5sigma=outliers),checks_scope='R necessary checks only; no repeat historical gates',archived_engineering_failures=[str(p) for p in (ROOT/'recoveries').glob('*/decision_prefix_assertion_*.json')],total_candidate_attempts=len((Path('/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/xr0/test_recovery_R_1004')/'candidate_decisions.jsonl').read_text().splitlines()),engineering_repairs_journal=str(ROOT/'engineering_repairs.jsonl'),selection_revisions_journal=str(ROOT/'candidate_selection_revisions.jsonl'),seed43_sources=[m['scene']['episode_id'] for m in mined if m['inference_seed']==43])
 write(ROOT/'data_audit.json',report)
 write(ROOT/'prepared_checks.json',dict(status='PASS',schedule_sha256=sha(schedule),manifest_sha256=sha(manifest),audit_sha256=sha(ROOT/'data_audit.json'),trajectory_sha256={d['annotation']:sha(d['annotation']) for d in accepted},video_sha256={v[0]['path']:sha(v[0]['path']) for d in accepted for v in inputs.recovery.episodes[d['annotation']]['observations'].values()},baseline=PROTOCOL['baseline']))
 print(json.dumps({k:v for k,v in report.items() if k in ['status','accepted_counts','candidate_counts','unique_source_episodes','schedule_counts','schedule_sha256','recovery_available_unique_windows','recovery_schedule_unique_windows']}),flush=True)
if __name__=='__main__':prepare()
