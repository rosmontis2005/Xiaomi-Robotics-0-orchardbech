#!/usr/bin/env python3
"""CPU-only frozen event metadata and auxiliary panels; never mutate prior artifacts."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,collections,math,datetime
import numpy as np
ROOT=Path(__file__).resolve().parent
XR0=ROOT.parent.parent
AB=XR0/'test_128_episode_AB_1003'
ORCHARD=Path('/home/rosmontis/Projects/orchardbench')
DATA=ORCHARD/'data/orchard_v1_2650/filtered'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
 p=ROOT/name
 with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def phase_at(t,indices):
 tr=t['orchardbench']['fixed_base_expert']['state_trace']; times=np.array([r['frame']/60 for r in tr]);ts=np.asarray(t['orchardbench']['timestamps'])
 return [tr[max(0,int(np.searchsorted(times,ts[i]+1e-7,side='right')-1))]['state'] for i in indices]
def events(ep,role):
 p=Path(ep['annotation']);t=load(p);assert sha(p)==ep['annotation_sha256']
 n=t['num_frames']; ts=np.asarray(t['orchardbench']['timestamps']);assert len(ts)==n and np.allclose(ts,np.arange(n)/30,atol=1e-7,rtol=0)
 tr=t['orchardbench']['fixed_base_expert']['state_trace'];pull=next(r for r in tr if r['state']=='PULL'); ev=pull['frame']/60
 assert abs(ev-pull['sim_time'])<=.000051 and pull['target_apple']==t['orchardbench']['fixed_base_expert']['selected_apple_debug_index']
 flags=(ts>=ev-.30-1e-7)&(ts<=ev+.10+1e-7)
 close=ep['closure_reference']['first_close_action_index'];ct=float(ts[min(close+1,n-1)])
 errors={}
 for k in ['ee_pos','ee_rotm','arm_joint','gripper_pos']:
  a=np.asarray(t['actions'][k]);p=np.asarray(t['proprios'][k]);err=float(np.max(np.abs(a[:-1]-p[1:])));assert err<1e-8,(ep['episode_id'],k,err);errors[k]=err
 return dict(split=ep['split'],episode_id=ep['episode_id'],seed=ep['seed'],role=role,annotation=ep['annotation'],annotation_sha256=ep['annotation_sha256'],num_frames=n,event_time_s=ev,event_source='first PULL state_trace entry physics_frame / 60; expert target association proxy, not student held truth',event_physics_frame=pull['frame'],expert_target_apple_id=pull['target_apple'],interval_start_s=ev-.30,interval_end_s=ev+.10,timestamps_s=ts.tolist(),contact_target_flags_by_observation_index=flags.tolist(),contact_observation_indices=np.flatnonzero(flags).tolist(),first_close_action_index=close,first_close_target_time_s=ct,first_close_minus_pull_s=ct-ev,first_close_inside_interval=bool(flags[min(close+1,n-1)]),event_recorded_frame=int(np.searchsorted(ts,ev-1e-7)),phase_events=[dict(phase=r['state'],time_s=r['frame']/60,recorded_frame=min(n-1,int(np.searchsorted(ts,r['frame']/60-1e-7)))) for r in tr],source_next_state_action_max_error=errors),t

def geometry(m):
 tcp=np.asarray(m['state_trace'][0]['tcp_world']);fruit=np.asarray(m['stance']['selected_apple_initial_world_pose']);d=fruit-tcp;distance=float(np.linalg.norm(d));angle=math.atan2(d[1],d[0]);pixels=max(m['target_visibility']['initial']['static_visible_pixels'],m['target_visibility']['initial']['wrist_visible_pixels'])
 return dict(target_world=fruit.tolist(),reset_tcp_world_rounded=tcp.tolist(),tcp_to_target_distance_m=distance,approach_world_bearing_rad=angle,target_height_m=float(fruit[2]),initial_best_view_pixels=pixels,selection_features=[math.cos(angle),math.sin(angle),distance,float(fruit[2]),math.log1p(pixels)])
def diverse(candidates,n,scale):
 lo=np.asarray(scale['robust_low_p05']);hi=np.asarray(scale['robust_high_p95']);mat=np.array([np.clip((np.asarray(r['geometry']['selection_features'])-(lo+hi)/2)/np.maximum(hi-lo,1e-9),-1,1) for r in candidates]);center=np.median(mat,axis=0);first=min(range(len(candidates)),key=lambda i:(float(np.sum((mat[i]-center)**2)),candidates[i]['seed']));chosen=[];dist=np.sum((mat-mat[first])**2,axis=1)
 for k in range(n):
  i=first if k==0 else min(range(len(candidates)),key=lambda i:(-float(dist[i]),candidates[i]['seed']));chosen.append(candidates[i]);dist=np.minimum(dist,np.sum((mat-mat[i])**2,axis=1));dist[i]=-np.inf
 return chosen

def main():
 s=load(AB/'selection.json');m0=load(s['source_m0_selection']);old={e['episode_id']:e for e in m0['episodes']}
 eps=[(e,'train') for e in s['episodes_train']]+[(e,'old_val') for e in m0['episodes'] if e['split']=='val']+[(e,'new_val') for e in s['episodes_new_val']]
 index={};traj={}
 for ep,role in eps:
  key=ep['split']+'/'+ep['episode_id'];e,t=events(ep,role);index[key]=e;traj[key]=t
 schedule=[json.loads(l) for l in (AB/'schedules/B.jsonl').read_text().splitlines()];assert len(schedule)==8000
 provenance=dict(old_selection_path=str(AB/'selection.json'),old_selection_sha256=sha(AB/'selection.json'),B_schedule_path=str(AB/'schedules/B.jsonl'),B_schedule_sha256=sha(AB/'schedules/B.jsonl'),stats_path=s['stats_path'],stats_sha256=s['stats_sha256'],script_sha256=sha(__file__))
 save('event_index.json',dict(schema='orchard_contact_event_index_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),provenance=provenance,definition=dict(proxy='first expert PULL transition, NOT measured student grasp',pre_seconds=.30,post_seconds=.10,time_tolerance_s=1e-7,action_target_index='min(window_frame + horizon_zero_based + 1, num_frames - 1)',raw_contact_weight=2,raw_other_weight=1,active_action_dims=list(range(7)),labels_are_model_inputs=False),episodes=index))
 stat=collections.defaultdict(collections.Counter);phase=collections.Counter();weighted=collections.Counter();near_total=0;hitwindows=0
 with (ROOT/'L_schedule_weights.jsonl').open('x') as f:
  for k,r in enumerate(schedule,1):
   assert r['step']==k and r['split']=='train';key='train/'+r['episode_id'];e=index[key];ts=e['timestamps_s'];inds=np.minimum(np.arange(r['frame']+1,r['frame']+31),e['num_frames']-1);flags=[e['contact_target_flags_by_observation_index'][int(i)] for i in inds];raw=[2. if v else 1. for v in flags];mean=np.mean(raw);normalized=np.array(raw)/mean
   row=dict(step=k,episode_id=r['episode_id'],split='train',frame=r['frame'],window_id=r['window_id'],annotation_sha256=r['annotation_sha256'],target_indices=inds.tolist(),contact_flags=flags,raw_target_weights=raw,event_time_s=e['event_time_s']);f.write(json.dumps(row,separators=(',',':'))+'\n')
   n=sum(flags);near_total+=n;hitwindows+=n>0;stat[r['group']]['windows']+=1;stat[r['group']]['contact_targets']+=n;stat[r['group']]['all_targets']+=30;stat[r['group']]['effective_contact_fraction_sum']+=float(np.sum(normalized[np.array(flags)])/30)
   for lab,w in zip(phase_at(traj[key],inds),normalized):phase[lab]+=1;weighted[lab]+=float(w)
 panel=[];oldkeys={(w['split'],w['episode_id'],w['frame']) for rows in s['evaluation_sets'].values() for w in rows};omissions=[]
 peps=[(e,'old_train') for e in m0['episodes'] if e['split']=='train']+[(e,'old_val') for e in m0['episodes'] if e['split']=='val']+[(e,'new_val') for e in s['episodes_new_val']]
 for ep,cohort in peps:
  key=ep['split']+'/'+ep['episode_id'];e=index[key];t=traj[key];n=e['num_frames'];p=e['event_recorded_frame'];d=next(ev['recorded_frame'] for ev in e['phase_events'] if ev['phase']=='DROP');used=set()
  desired=[('near_contact','event_h10',p-10),('near_contact','event_h5',p-5),('near_contact','event_h1',p-1),('late_release','drop_h15',d-15),('late_release','drop_entry',d),('late_release','last_full_window',n-30)]
  for kind,label,frame in desired:
   if frame<0 or frame>n-30 or frame in used:omissions.append(dict(episode_id=e['episode_id'],label=label,requested_frame=frame,reason='invalid_or_duplicate'));continue
   used.add(frame);inds=np.minimum(np.arange(frame+1,frame+31),n-1);panel.append(dict(window_id=f"{ep['split']}/{ep['episode_id']}/frame{frame:04d}",split=ep['split'],episode_id=ep['episode_id'],seed=ep['seed'],annotation=ep['annotation'],annotation_sha256=ep['annotation_sha256'],frame=frame,time_s=e['timestamps_s'][frame],label=label,labels=[label],evaluation_set=kind+'_'+cohort,cohort=cohort,panel_kind=kind,anchor_phase=phase_at(t,[frame])[0],target_phase_by_horizon=phase_at(t,inds),target_times_s=[e['timestamps_s'][int(i)] for i in inds],contact_flags=[e['contact_target_flags_by_observation_index'][int(i)] for i in inds],event_time_s=e['event_time_s'],overlaps_original_240=(ep['split'],ep['episode_id'],frame) in oldkeys,selection_rule='fixed event landmarks; no checkpoint outputs consulted'))
 save('auxiliary_panel.json',dict(schema='orchard_contact_late_panel_v1',provenance=provenance,original_240_unchanged_reference=str(AB/'eval_cache.pt'),use='independent diagnostics only; NEVER optimizer data additions or checkpoint selection',windows=panel,omissions=omissions,counts=dict(collections.Counter(w['evaluation_set'] for w in panel)),original_overlap_count=sum(w['overlaps_original_240'] for w in panel),target_phase_counts=dict(collections.Counter(p for w in panel for p in w['target_phase_by_horizon']))))
 excluded={e['episode_id'] for e in s['episodes_train']+s['episodes_new_val']+s['episodes_heldout']+m0['episodes']}|set(s['selection_rule']['excluded_previous_val_ids'])|set(s['selection_rule']['excluded_visual_train_ids']);manifest=[json.loads(l) for l in (DATA/'manifest.jsonl').read_text().splitlines()];cand=[]
 for m in manifest:
  if m['split']!='val' or m['episode_id'] in excluded:continue
  if not(m['accepted'] and m['oracle_strict_success'] and m['replay']['IK_failed_steps']==0 and m['replay']['dwell_timeout_count']==0):continue
  g=geometry(m)
  if g['initial_best_view_pixels']<128:continue
  cand.append(dict(split='val',episode_id=m['episode_id'],seed=m['seed'],annotation=str(DATA/m['annotation']),annotation_sha256=m['annotation_sha256'],geometry=g,visibility=m['target_visibility'],replay=m['replay'],oracle_strict_success=True))
 held=diverse(cand,24,m0['selection_rule']['geometric_spread']['val']);assert len({e['seed'] for e in held})==24
 for e in held:assert sha(e['annotation'])==e['annotation_sha256'] and load(e['annotation'])['seed']==e['seed']
 save('fresh24_heldout.json',dict(schema='orchard_fresh24_frozen_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_manifest_path=str(DATA/'manifest.jsonl'),source_manifest_sha256=sha(DATA/'manifest.jsonl'),provenance=provenance,selection_rule='val only; accepted, historical strict replay success, IK=0, dwell=0, best initial raw pixels>=128; deterministic geometry farthest point with old fixed M0 val robust scales; no model outcomes consulted',excluded_ids=sorted(excluded),eligible_candidates=len(cand),scenes=held,model_seeds=[42,43],use='final selected candidate vs B8000 only; no training, no offline panel, no development tuning',limitations='Historical curated observable validation scenes, not new procedurally generated seeds. Current-path GT and actual prepared-image review pending; no automatic failure exclusion or scene replacement.'))
 # Select 24 train episodes spanning event/closure discrepancy without using model outcomes.
 train=[e for e in index.values() if e['split']=='train'];sort=sorted(train,key=lambda e:(e['first_close_minus_pull_s'],e['seed']));sample=[sort[int(i)] for i in np.linspace(0,len(sort)-1,24,dtype=int)]
 save('review_24_manifest.json',dict(schema='orchard_event_proxy_review_v1',rule='24 unique train scenes at evenly-spaced ranks of close-target minus PULL time, seed tie break; no model results',episodes=[dict(episode_id=e['episode_id'],annotation=e['annotation'],annotation_sha256=e['annotation_sha256'],event_time_s=e['event_time_s'],first_close_minus_pull_s=e['first_close_minus_pull_s'],frames=[e['contact_observation_indices'][0],e['event_recorded_frame'],e['contact_observation_indices'][-1]]) for e in sample],review_status='PENDING actual prepared RGB sheets and root review'))
 audit=dict(status='PASS_CPU_METADATA',provenance=provenance,event_index_sha256=sha(ROOT/'event_index.json'),schedule_weights_sha256=sha(ROOT/'L_schedule_weights.jsonl'),train_episodes=128,all_event_episodes=len(index),schedule_steps=8000,schedule_source_unchanged=True,contact_windows=hitwindows,contact_targets=near_total,total_targets=8000*30,raw_contact_target_fraction=near_total/(8000*30),normalized_effective_contact_fraction=sum(v['effective_contact_fraction_sum'] for v in stat.values())/8000,per_group={g:dict(v) for g,v in stat.items()},unweighted_phase_counts=dict(phase),normalized_weighted_phase_mass=dict(weighted),first_close_target_in_event_count=sum(e['first_close_inside_interval'] for e in train),close_to_pull_range_s=[min(e['first_close_minus_pull_s'] for e in train),max(e['first_close_minus_pull_s'] for e in train)],source_next_state_actions_exact=True,aux_panel_counts=dict(collections.Counter(w['evaluation_set'] for w in panel)),aux_overlap_old240=sum(w['overlaps_original_240'] for w in panel),aux_omissions=omissions,fresh_heldout=24,fresh_heldout_eligible=len(cand),gpu_used=False)
 save('event_preflight.json',audit);print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
