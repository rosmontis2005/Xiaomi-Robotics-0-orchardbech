"""R retry2: sparse teacher bridge windows, explicitly separate from student onsets."""
import bootstrap
import json,copy
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
from recovery_common import *
from recovery_dataset import ScheduledInputs
from treesim.orchard_action import encode_window,decode_targets,EPS
ORIGINAL=ROOT.parents[1];PREVIOUS=ROOT.parent/'version_02';QUOTAS=dict(reset=560,first1_4=280,REACH=420,GRASP=700,PULL=420,TRANSPORT=280,DROP=140);COUNTS=dict(original_B=2800,R1=500,R2=200,R3=200,R_continuation_teacher=300);BRIDGE=dict(transport_to_drop=150,DROP_transition=100,DONE_settle=50)
def draw_episode_balanced(rng,options,quota,reserve_frame0=False):
 groups=defaultdict(list)
 for d,frames in options:
  for f in frames:groups[d['source_episode_id']].append((d,f))
 sources=sorted(groups);assert sources;base,extra=divmod(quota,len(sources));draws=np.full(len(sources),base,dtype=int);draws[rng.permutation(len(sources))[:extra]]+=1;chosen=[]
 for source,n in zip(sources,draws):
  pool=groups[source];picked=[]
  if reserve_frame0:
   unique={d['candidate_id']:d for d,f in pool if f==0};picked=[(d,0) for d in unique.values()]
  assert n>=len(picked);picked += [pool[int(i)] for i in rng.choice(len(pool),size=int(n)-len(picked),replace=True)];chosen += picked
 return chosen

def prepare():
 protection_check();original_audit=json.loads((ORIGINAL/'data_audit.json').read_text());accepted=[json.loads(l) for l in (PREVIOUS/'recovery_manifest.jsonl').read_text().splitlines()];assert Counter(d['category'] for d in accepted)==dict(R1=12,R2=6,R3=6)
 trajectories={};bridge={key:[] for key in BRIDGE}
 for d in accepted:
  assert sha(d['annotation'])==d['annotation_sha256'];t=json.loads(Path(d['annotation']).read_text());trajectories[d['annotation']]=t;r=t['recovery'];assert r['student_checkpoint_sha256']==PROTOCOL['baseline']['sha256'];assert r['source_episode_id'] not in PROTOCOL['protected_episode_ids'];assert r['source_scene_seed'] not in PROTOCOL['protected_scene_seeds'];assert r['teacher_final_result']=='strict_success' and r['oracle_replay_result']['status']=='PASS' and r['initialization_audit']['physics_unchanged']
  phase=t['orchardbench']['phase_by_frame'];drop=phase.index('DROP');done=phase.index('DONE');last=t['num_frames']-30;d['takeover_anchors']=list(d['legal_anchors']);assert 0 in d['takeover_anchors'] and all(f<=29 for f in d['takeover_anchors'])
  groups=dict(transport_to_drop=sorted({drop-29,drop-15,drop-1}),DROP_transition=[drop,drop+5,drop+10],DONE_settle=sorted({done,min(done+5,last)}))
  for group,frames in groups.items():
   frames=[f for f in frames if 30<=f<=last];assert frames,(d['candidate_id'],group)
   assert all(phase[f]=={'transport_to_drop':'TRANSPORT','DROP_transition':'DROP','DONE_settle':'DONE'}[group] for f in frames)
   bridge[group].append((d,frames));groups[group]=frames
  d['teacher_continuation_anchors_by_group']=groups;d['legal_anchors']=sorted(set(d['takeover_anchors'])|{f for fs in groups.values() for f in fs});d['source_schema_note']='Teacher-continuation anchors are not new student takeover observations; original takeover control_step metadata remains original.'
 manifest=ROOT/'recovery_manifest.jsonl'
 with manifest.open('x') as stream:
  for d in accepted:stream.write(json.dumps(d)+'\n')
 rng=np.random.default_rng(42);selection=json.loads((AB/'selection.json').read_text());B=[]
 # Preserve the existing B group/episode balanced sampler, scaling only quotas.
 for group,quota in QUOTAS.items():
  choices=[(ep,ep['group_frames'][group]) for ep in selection['episodes_train'] if ep['group_frames'][group]];base,extra=divmod(quota,len(choices));draws=np.full(len(choices),base,dtype=int);draws[rng.permutation(len(choices))[:extra]]+=1
  for (ep,frames),n in zip(choices,draws):
   for f in rng.choice(frames,size=int(n),replace=True):B.append(dict(source='original_B',group=group,episode_id=ep['episode_id'],source_episode_id=ep['episode_id'],seed=ep['seed'],split='train',frame=int(f),annotation=ep['annotation'],annotation_sha256=ep['annotation_sha256'],window_id=f"original_B/{ep['episode_id']}/frame{int(f):04d}"))
 B=[B[int(i)] for i in rng.permutation(len(B))];recovery=[]
 def record(d,f,source,group):
  return dict(source=source,group=group,episode_id=d['candidate_id'],source_episode_id=d['source_episode_id'],seed=trajectories[d['annotation']]['seed'],split='train',frame=int(f),annotation=d['annotation'],annotation_sha256=d['annotation_sha256'],window_id=f"{d['category']}/{d['candidate_id']}/frame{int(f):04d}",observation_start_role='student_takeover_neighbourhood' if source!='R_continuation_teacher' else 'teacher_continuation',recovery_origin_category=d['category'])
 for cat in ['R1','R2','R3']:
  options=[(d,d['takeover_anchors']) for d in accepted if d['category']==cat]
  for d,f in draw_episode_balanced(rng,options,COUNTS[cat],True):recovery.append(record(d,f,cat,cat))
 for group,quota in BRIDGE.items():
  for d,f in draw_episode_balanced(rng,bridge[group],quota):recovery.append(record(d,f,'R_continuation_teacher',group))
 rows=B+recovery;rows=[rows[int(i)] for i in rng.permutation(len(rows))]
 for i,row in enumerate(rows,1):row['step']=i
 assert len(rows)==4000 and Counter(r['source'] for r in rows)==COUNTS
 schedule=ROOT/'schedule_R.jsonl'
 with schedule.open('x') as stream:
  for row in rows:stream.write(json.dumps(row)+'\n')
 inputs=ScheduledInputs(rows);contract={}
 for source in COUNTS:
  i=next(i for i,r in enumerate(rows) if r['source']==source);sample=inputs.sample(rows[i]);batch=inputs.get(i);assert set(sample)=={'messages','action','action_mask','state'};assert tuple(sample['action'].shape)==(30,32) and tuple(sample['state'].shape)==(1,32) and torch.isfinite(sample['action']).all() and torch.all(sample['action'][:,7:]==0);contract[source]={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in batch.items()}
 assert all(x==contract['original_B'] for x in contract.values())
 for d in accepted:
  t=inputs.recovery.episodes[d['annotation']];changed=copy.deepcopy(t);changed['recovery']={'poison':'NOT POLICY INPUT'};changed['orchardbench']={'poison':999};frame=d['teacher_continuation_anchors_by_group']['transport_to_drop'][0];a=inputs.dm.collate_fn([inputs.recovery.from_trajectory(t,frame)]);b=inputs.dm.collate_fn([inputs.recovery.from_trajectory(changed,frame)]);assert set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
 mean=inputs.recovery.mean;std=inputs.recovery.std;z=[];outliers=[];phase_cache={}
 for d in accepted:
  t=trajectories[d['annotation']];p=t['proprios'];actions=t['actions'];assert np.allclose(np.diff(t['orchardbench']['timestamps']),1/30,atol=1e-9,rtol=0)
  for key in actions:assert np.array_equal(actions[key],p[key][1:]+p[key][-1:])
  for frame in d['legal_anchors']:
   action=encode_window(t,frame);xyz,rot,width=decode_targets(action,p['ee_pos'][frame],p['ee_rotm'][frame]);sl=slice(frame,frame+30);assert np.allclose(xyz,actions['ee_pos'][sl],atol=2e-6,rtol=0) and np.allclose(rot,np.asarray(actions['ee_rotm'][sl]).reshape(-1,3,3),atol=2e-6,rtol=0) and np.allclose(width,np.asarray(actions['gripper_pos'][sl]).ravel(),atol=2e-6,rtol=0);value=(action-mean)/(std+EPS);z.append(value[:,:7]);maximum=float(np.abs(value[:,:7]).max())
   if maximum>5:outliers.append(dict(trajectory=d['candidate_id'],frame=frame,max_abs_normalized=maximum))
  phase_cache[d['annotation']]=t['orchardbench']['phase_by_frame']
 anchor=defaultdict(Counter);future=defaultdict(Counter)
 for row in rows:
  if row['annotation'] not in phase_cache:
   t=json.loads(Path(row['annotation']).read_text());times=t['orchardbench']['timestamps'];trace=t['orchardbench']['fixed_base_expert']['state_trace'];events=np.array([r['frame']/60 for r in trace]);idx=np.searchsorted(events,np.asarray(times)+1e-7,side='right')-1;phase_cache[row['annotation']]=[trace[max(0,int(i))]['state'] for i in idx]
  phase=phase_cache[row['annotation']];anchor[row['source']].update([phase[row['frame']]]);future[row['source']].update(phase[min(i,len(phase)-1)] for i in range(row['frame']+1,row['frame']+31))
 exposure=Counter(r['window_id'] for r in recovery);z=np.concatenate(z,axis=0)
 report=dict(status='PASS',candidate_counts=original_audit['candidate_counts'],accepted_counts=dict(Counter(d['category'] for d in accepted)),rejected_counts=original_audit['rejected_counts'],unique_source_episodes=len({d['source_episode_id'] for d in accepted}),recovery_available_unique_windows=sum(len(d['legal_anchors']) for d in accepted),true_takeover_available_windows=sum(len(d['takeover_anchors']) for d in accepted),teacher_continuation_available_windows=sum(sum(len(fs) for fs in d['teacher_continuation_anchors_by_group'].values()) for d in accepted),recovery_schedule_unique_windows=len(exposure),repeat_draws=len(recovery)-len(exposure),schedule_counts=dict(Counter(r['source'] for r in rows)),original_B_groups=dict(Counter(r['group'] for r in B)),teacher_continuation_groups=dict(Counter(r['group'] for r in recovery if r['source']=='R_continuation_teacher')),schedule_sha256=sha(schedule),manifest_sha256=sha(manifest),schedule_seed=42,actual_anchor_phase_counts_schedule={k:dict(v) for k,v in anchor.items()},future_target_phase_counts_schedule={k:dict(v) for k,v in future.items()},trajectory_exposure=dict(Counter(r['episode_id'] for r in recovery)),source_exposure_by_category={s:dict(Counter(r['source_episode_id'] for r in recovery if r['source']==s)) for s in COUNTS if s!='original_B'},window_exposure=dict(exposure),tensor_contract=contract,privileged_metadata_mutation_invariance='PASS',next_state_indexing='PASS',roundtrip='PASS',teacher_and_oracle_all_accepted='PASS (existing full trajectories replayed, labels unchanged)',new_onpolicy_candidates_used=0,new_onpolicy_candidate_rejects=4,parent_data_audit_sha256=sha(ORIGINAL/'data_audit.json'),normalization=dict(path=str(STATS),sha256=sha(STATS),recomputed=False,labels_clipped=False,max_abs_z_by_active_dim=np.abs(z).max(axis=0),outliers_over_5sigma=outliers),sampling_scope='Only8 declared sparse bridge anchors per trajectory plus first30 onsets. Teacher continuation is separately counted; never labelled as student deviations.')
 write(ROOT/'data_audit.json',report);write(ROOT/'prepared_checks.json',dict(status='PASS',schedule_sha256=sha(schedule),manifest_sha256=sha(manifest),audit_sha256=sha(ROOT/'data_audit.json'),trajectory_sha256={d['annotation']:sha(d['annotation']) for d in accepted},video_sha256={v[0]['path']:sha(v[0]['path']) for d in accepted for v in trajectories[d['annotation']]['observations'].values()},baseline=PROTOCOL['baseline']));print(json.dumps({k:report[k] for k in ['status','schedule_counts','recovery_available_unique_windows','recovery_schedule_unique_windows','true_takeover_available_windows','teacher_continuation_available_windows','teacher_continuation_groups','schedule_sha256']}))
if __name__=='__main__':prepare()
