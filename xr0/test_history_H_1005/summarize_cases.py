"""Read-only summaries of existing boundary observers; never used by policy."""
import json,csv,sys
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent;XR0=ROOT.parent

def load(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

def summarize_case(folder,cohort,inference_seed):
 folder=Path(folder);s=load(folder/'summary.json');m=load(folder/'metrics.json');steps=lines(folder/'steps.jsonl');initial=load(folder/'initial.json');stance=initial['info']['reset_stance_debug'];planned=int(stance['apple_index']);xy=np.array(stance['base_xy']);yaw=stance['base_yaw'];cr=Rotation.from_euler('z',yaw);cp=np.r_[xy,0.];bucket=cp+cr.apply([-.26,0.,.46]);chains=s['strict_placement']['fruit_chains'];qualified={int(k):v for k,v in chains.items() if v['held15_step'] is not None};events=s['strict_placement']['release_events'];release_info=[];distances=[];actual_held_distances=[]
 for r in steps:
  snap=r['grasp_after'];apple=snap.get('planned_apple');held=snap['held_apple_id']
  if held is not None:
   held_geo=snap.get('fruit_bucket_geometry',{}).get(str(held))
   held_obs=snap.get('planned_apple') if held==planned else next((x for x in [snap.get('nearest_tcp_apple'),snap.get('nearest_palm_center_apple'),*snap.get('contact_apples',[])] if x and x['apple_id']==held),None)
   if held_geo:actual_held_distances.append((r['control_step'],float(held_geo['distance_m'])))
   elif held_obs:
    geo=snap.get('placement_geometry');actual_held_distances.append((r['control_step'],float(geo['planned_fruit_bucket_distance_m']) if geo and held==planned else float(np.linalg.norm(np.asarray(held_obs['world_position'])-bucket))))
  if apple is not None:
   world=np.array(apple['world_position']);geo=snap.get('placement_geometry');distance=float(geo['planned_fruit_bucket_distance_m']) if geo else float(np.linalg.norm(world-bucket));local=np.array(geo['planned_fruit_base_local']) if geo else cr.inv().apply(world-cp)
   distances.append((r['control_step'],distance,snap['held_apple_id']==planned))
 for event in events:
  r=steps[event['step']-1];snap=r['grasp_after'];apple=snap.get('planned_apple') if event['apple_id']==planned else next((x for x in [snap.get('nearest_tcp_apple'),snap.get('nearest_palm_center_apple'),*snap.get('contact_apples',[])] if x and x['apple_id']==event['apple_id']),None)
  distance=local=None
  if apple:
   world=np.array(apple['world_position']);geo=snap.get('placement_geometry');distance=float(geo['planned_fruit_bucket_distance_m']) if geo and event['apple_id']==planned else float(np.linalg.norm(world-bucket));local=np.array(geo['planned_fruit_base_local']) if geo and event['apple_id']==planned else cr.inv().apply(world-cp)
  event_geo=snap.get('fruit_bucket_geometry',{}).get(str(event['apple_id']))
  if event_geo:distance=float(event_geo['distance_m']);local=np.asarray(event_geo['base_local'])
  release_info.append(dict(**event,fruit_bucket_distance_m=distance,within_bucket_xy=None if local is None else bool(abs(local[0]+.26)<.15 and abs(local[1])<.15),in_bucket_at_release=event['apple_id'] in r['fruit_state']['in_bucket_apple_ids'],reasonable_region=None if distance is None else bool(distance<=.15 and abs(local[0]+.26)<.15 and abs(local[1])<.15),spatial_diagnostic_only=True))
 held_dist=actual_held_distances;min_held=min((d for _,d in held_dist),default=None);allmin=min((d for _,d,_ in distances),default=None);maxstable=max((v['max_stable_bucket_steps'] for v in chains.values()),default=0);h=bool(qualified);detach=any(v['first_detach_after_grasp_step'] is not None for v in qualified.values());release=bool(events);valid_release=any(e['valid_chain'] for e in events);reasonable=any(e['valid_chain'] and e['reasonable_region'] is True for e in release_info);branches=s['max_branch_break_count'];incidental=any(len(r['fruit_state']['detached_apple_ids'])>1 for r in steps);last100=[d for step,d in held_dist if step>steps[-1]['control_step']-100];long_stall=bool(h and not valid_release and len(last100)>=90 and min(last100)>.15 and max(last100)-min(last100)<.03)
 if s['strict_success']:failure='strict_success'
 elif branches or incidental:failure='physical_anomaly'
 elif not h:failure='no_stable_grasp'
 elif not detach:failure='held_not_detached'
 elif release and not reasonable:failure='release_far_or_outside_bucket'
 elif release:failure='release_without_bucket60'
 elif min_held is not None and min_held<=.15:failure='bucket_approach_no_release'
 elif long_stall:failure='held_transport_stall'
 else:failure='transport_incomplete_no_release'
 postheld=max((s['control_steps']-v['held15_step'] for v in qualified.values()),default=0)
 return dict(post_held_steps=postheld,held_no_bucket_over300=bool(postheld>=300 and min_held is not None and min_held>.15),model=s['checkpoint'],cohort=cohort,seed=s['seed'],inference_seed=inference_seed,split=s['split'],episode_id=s['episode_id'],held15=h,same_fruit_detach=detach,release=release,valid_chain_release=valid_release,reasonable_release=reasonable,strict=bool(s['strict_success']),planned_apple_id=planned,min_planned_fruit_bucket_distance_m=allmin,min_held_fruit_bucket_distance_m=min_held,min_held_planned_fruit_bucket_distance_m=min((d for _,d,isheld in distances if isheld),default=None),bucket_neighborhood=bool(min_held is not None and min_held<=.15),max_stable_bucket_steps=maxstable,long_held_stall=long_stall,branch_break=bool(branches),physical_anomaly=bool(branches or incidental),failure=failure,release_events=release_info,held_chains=chains,control_steps=s['control_steps'],replans=s['replans'],directory=str(folder),historical_geometry_note='Bucket calculated from reset fixed-base pose if per-step geometry absent; no policy input.')

def aggregate(cases):
 groups={}
 for cohort in ['dev8','dev8_train2','dev8_val6','historical_seed42_43']:
  subset=[c for c in cases if c['cohort']==cohort or cohort=='dev8_train2' and c['cohort']=='dev8' and c['split']=='train' or cohort=='dev8_val6' and c['cohort']=='dev8' and c['split']=='val']
  if not subset:continue
  groups[cohort]=dict(n=len(subset),**{key:sum(bool(c[key]) for c in subset) for key in ['held15','same_fruit_detach','release','valid_chain_release','reasonable_release','strict','bucket_neighborhood','long_held_stall','branch_break','physical_anomaly']},max_stable_bucket_steps=max(c['max_stable_bucket_steps'] for c in subset),failures=dict(Counter(c['failure'] for c in subset)),min_held_bucket_distance_m={str(c['seed'])+f"/rng{c['inference_seed']}":c['min_held_fruit_bucket_distance_m'] for c in subset})
 return groups

def summarize_evaluation(evaluation,baseline=False):
 evaluation=Path(evaluation);p=load(evaluation/'protocol.json');cases=[]
 for j in p['jobs']:
  folder=XR0/'test_grasp_1004/evaluation/rollouts'/j['baseline_job_id'] if baseline else evaluation/'rollouts'/j['job_id']
  cohort='historical_seed42_43' if j['stage'].startswith('regression') else 'dev8';cases.append(summarize_case(folder,cohort,j['inference_seed']))
 return cases,aggregate(cases)

if __name__=='__main__':
 evaluation=Path(sys.argv[1]);cases,groups=summarize_evaluation(evaluation);output=Path(sys.argv[2]) if len(sys.argv)>2 else evaluation
 assert output.resolve().is_relative_to(ROOT);output.mkdir(parents=True,exist_ok=True)
 (output/'cases.json').write_text(json.dumps(cases,indent=2));(output/'aggregate.json').write_text(json.dumps(groups,indent=2))
 keys=['model','cohort','seed','inference_seed','split','held15','same_fruit_detach','release','valid_chain_release','reasonable_release','strict','min_held_planned_fruit_bucket_distance_m','bucket_neighborhood','max_stable_bucket_steps','long_held_stall','branch_break','physical_anomaly','failure']
 with (output/'cases.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(cases)
 print(json.dumps(groups))
