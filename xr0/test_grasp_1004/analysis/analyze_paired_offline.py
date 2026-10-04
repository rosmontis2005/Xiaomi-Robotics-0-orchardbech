#!/usr/bin/env python3
"""CPU-only, read-only model artifact analysis. Writes only beneath this analysis dir."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,collections,csv,datetime,hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent;EXPERIMENT=ROOT.parent;DATA=EXPERIMENT/'data';AB=EXPERIMENT.parent/'test_128_episode_AB_1003'
METRICS=('position_error_m','rotation_error_rad','width_error_m');BANDS={'first1':1,'first5':5,'first10':10,'first30':30};PHASES=('REACH','GRASP','PULL','TRANSPORT','DROP','DONE')

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
 p=Path(p).resolve();assert p.is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def key(w):return (w['split'],w['episode_id'],int(w['frame']))
def filename(w):return f"{w['split']}_{w['episode_id']}_frame{w['frame']:04d}.npz"
def metadata():
 selection=load(AB/'selection.json');event=load(DATA/'event_index.json')['episodes'];old=[]
 for cohort,windows in selection['evaluation_sets'].items():
  for w in windows:
   m=dict(w);m.update(cohort=cohort,panel_kind='original',evaluation_set=cohort);old.append(m)
 aux=load(DATA/'auxiliary_panel.json')['windows']
 for w in old+aux:
  e=event[w['split']+'/'+w['episode_id']];ix=np.minimum(np.arange(w['frame']+1,w['frame']+31),e['num_frames']-1);w['contact_flags']=np.asarray([e['contact_target_flags_by_observation_index'][int(i)] for i in ix],dtype=bool);w['target_times_s']=np.asarray([e['timestamps_s'][int(i)] for i in ix]);w['target_phase_by_horizon']=np.asarray(w['target_phase_by_horizon']);w['event_time_s']=e['event_time_s']
 assert len(old)==240 and len(aux)==144;assert len(set(map(key,old)))==240 and len(set(map(key,aux)))==144;assert len(set(map(key,old))&set(map(key,aux)))==49
 return old,aux

def read_arrays(path,meta):
 with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 seeds=a['prediction_seeds'].astype(int);assert seeds.shape==(3,) and len(set(seeds))==3 and set(seeds)=={42,43,44};order=np.argsort(seeds)
 assert np.array_equal(a['horizon'],np.arange(1,31));assert np.array_equal(a['target_phase'],meta['target_phase_by_horizon']);assert np.allclose(a['target_time_s'],meta['target_times_s'],rtol=0,atol=1e-10)
 pe=np.linalg.norm(a['predicted_world_position']-a['gt_world_position'][None],axis=-1)
 re=Rotation.from_matrix((a['gt_world_rotation'][None].transpose(0,1,3,2)@a['predicted_world_rotation']).reshape(-1,3,3)).magnitude().reshape(3,30)
 we=np.abs(a['predicted_width_raw_m']-a['gt_width_m'][None])
 for m,expected in zip(METRICS,(pe,re,we)):
  assert a[m].shape==(3,30) and np.isfinite(a[m]).all();assert np.allclose(a[m],expected,atol=1e-8,rtol=1e-7),(path,m)
 return dict(seeds=seeds[order],errors={m:a[m][order] for m in METRICS},gt_action=a['gt_action_physical'],gt_world_position=a['gt_world_position'],gt_world_rotation=a['gt_world_rotation'],target_phase=a['target_phase'],target_time=a['target_time_s'],pred_width_intent=a['predicted_width_intent_proxy'][order],gt_width_intent=a['gt_width_intent_proxy'],all_arrays=a,path=str(path),sha256=sha(path))

def load_pair(bdir,ldir,metas,step=8000):
 bdir=Path(bdir);ldir=Path(ldir);reports=[load(d/'report.json') for d in (bdir,ldir)]
 for report,d in zip(reports,(bdir,ldir)):
  assert report['step']==step and report['windows']==len(metas) and report['generated_from_zero_action'] and not report['ground_truth_prefix_used'];assert report['seeds']==[42,43,44];assert report['selection_sha256']==sha(AB/'selection.json');assert {p.name for p in d.glob('*.npz')}=={filename(m) for m in metas}
 assert reports[0]['stats_sha256']==reports[1]['stats_sha256'] and reports[0]['model_num_steps']==reports[1]['model_num_steps']
 rowlookups=[{key(r):r for r in report['window_metrics']} for report in reports];records=[]
 for m in metas:
  b=read_arrays(bdir/filename(m),m);l=read_arrays(ldir/filename(m),m);assert np.array_equal(b['seeds'],l['seeds'])
  for k in ('gt_action','gt_world_position','gt_world_rotation','target_phase','target_time','gt_width_intent'):assert np.array_equal(b[k],l[k]),(filename(m),k)
  records.append(dict(meta=m,B=b,L=l,closures_B=rowlookups[0][key(m)]['closure_proxy'],closures_L=rowlookups[1][key(m)]['closure_proxy']))
 return records,dict(B_report=str(bdir/'report.json'),L_report=str(ldir/'report.json'),B_report_sha256=sha(bdir/'report.json'),L_report_sha256=sha(ldir/'report.json'),npz_pairs=len(records),GT_and_seed_pairing_exact=True)

def distribution(a):
 a=np.asarray(a,dtype=float);return dict(count=int(a.size),mean=float(a.mean()),p50=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max()))

def summarize(records,mask_fn,bootstrap_draws=2000):
 byscene=collections.defaultdict(lambda:collections.defaultdict(list));bs={m:[] for m in METRICS};ls={m:[] for m in METRICS};window_count=0;unique_target_count=0;windowrows=[]
 for r in records:
  mask=np.asarray(mask_fn(r['meta']),dtype=bool);assert mask.shape==(30,)
  if not mask.any():continue
  window_count+=1;unique_target_count+=int(mask.sum());scene=r['meta']['split']+'/'+r['meta']['episode_id'];wr=dict(window_id=r['meta']['window_id'],target_count=int(mask.sum()),paired_seed_count=3,metrics={})
  for metric in METRICS:
   b=r['B']['errors'][metric][:,mask];l=r['L']['errors'][metric][:,mask];bs[metric].append(b.ravel());ls[metric].append(l.ravel());byscene[scene][metric].append((b.ravel(),l.ravel()));wr['metrics'][metric]=dict(B=float(b.mean()),L=float(l.mean()),L_minus_B=float((l-b).mean()))
  windowrows.append(wr)
 if window_count==0:return dict(status='NO_TARGETS',windows=0,scenes=0,unique_window_target_count=0,paired_seed_target_count=0,metrics={})
 result=dict(status='OK',windows=window_count,scenes=len(byscene),unique_window_target_count=unique_target_count,paired_seed_target_count=unique_target_count*3,metrics={},per_scene=[])
 scenevalues={m:[] for m in METRICS}
 for scene in sorted(byscene):
  row=dict(scene=scene,metrics={})
  for metric in METRICS:
   b=np.concatenate([x[0] for x in byscene[scene][metric]]);l=np.concatenate([x[1] for x in byscene[scene][metric]]);vals=dict(B=float(b.mean()),L=float(l.mean()),L_minus_B=float((l-b).mean()),paired_seed_target_count=int(len(b)));row['metrics'][metric]=vals;scenevalues[metric].append([vals['B'],vals['L'],vals['L_minus_B']])
  result['per_scene'].append(row)
 for metric in METRICS:
  b=np.concatenate(bs[metric]);l=np.concatenate(ls[metric]);d=l-b;s=np.asarray(scenevalues[metric]);rng=np.random.default_rng(20261004);boot=s[rng.integers(0,len(s),size=(bootstrap_draws,len(s))),2].mean(axis=1)
  result['metrics'][metric]=dict(B_pooled=distribution(b),L_pooled=distribution(l),paired_pooled_delta=distribution(d),B_scene_equal_mean=float(s[:,0].mean()),L_scene_equal_mean=float(s[:,1].mean()),L_minus_B_scene_equal_mean=float(s[:,2].mean()),improved_scenes=int((s[:,2]<-1e-12).sum()),tied_scenes=int((np.abs(s[:,2])<=1e-12).sum()),worsened_scenes=int((s[:,2]>1e-12).sum()),scene_resample_descriptive_CI95=np.quantile(boot,[.025,.975]).tolist())
 return result

def metric_groups(records,panel_name):
 outputs={};cohorts=['all','old_train','old_val','new_val','combined_val'];filters=['all_targets','contact_event','outside_contact']+list(PHASES)
 for cohort in cohorts:
  chosen=[r for r in records if cohort=='all' or r['meta']['cohort']==cohort or(cohort=='combined_val' and r['meta']['split']=='val')]
  for target in filters:
   for band,n in BANDS.items():
    def mask(m,target=target,n=n):
     h=np.arange(30)<n
     if target=='contact_event':h &=m['contact_flags']
     elif target=='outside_contact':h &=~m['contact_flags']
     elif target!='all_targets':h &=m['target_phase_by_horizon']==target
     return h
    outputs[f'{panel_name}/{cohort}/{target}/{band}']=summarize(chosen,mask)
 # Explicit observation landmarks. near_hN means PULL-entry reference at recorded target N, NOT a held/reach event.
 for near,source in [('near_h1','event_h1'),('near_h5','event_h5'),('near_h10','event_h10'),('observation_reset','reset'),('observation_transport_entry','transport_entry'),('observation_late_drop_h15','drop_h15'),('observation_drop_entry','drop_entry'),('observation_last_full_window','last_full_window')]:
  selected=[r for r in records if r['meta']['label']==source or source in r['meta'].get('labels',[]) or r['meta'].get('auxiliary_landmark')==source]
  if not selected:continue
  for cohort in cohorts:
   chosen=[r for r in selected if cohort=='all' or r['meta']['cohort']==cohort or(cohort=='combined_val' and r['meta']['split']=='val')]
   for target in ['all_targets','contact_event']:
    for band,n in BANDS.items():outputs[f'{panel_name}/{cohort}/{near}/{target}/{band}']=summarize(chosen,lambda m,n=n,target=target:(np.arange(30)<n)&(m['contact_flags'] if target=='contact_event' else np.ones(30,dtype=bool)))
 return outputs

def closure_summary(records):
 rows=[]
 for r in records:
  a={int(x['seed']):x for x in r['closures_B']};b={int(x['seed']):x for x in r['closures_L']};assert set(a)==set(b)=={42,43,44}
  for seed in sorted(a):
   x,y=a[seed],b[seed];assert x['gt_first_close_switch_horizon']==y['gt_first_close_switch_horizon'];rows.append(dict(window_id=r['meta']['window_id'],cohort=r['meta']['cohort'],seed=seed,gt_horizon=x['gt_first_close_switch_horizon'],B_predicted_horizon=x['predicted_first_close_switch_horizon'],L_predicted_horizon=y['predicted_first_close_switch_horizon'],B_offset=x['close_switch_offset_targets'],L_offset=y['close_switch_offset_targets'],B_missing=x['missing_gt_close_switch'],L_missing=y['missing_gt_close_switch'],B_intent_mismatch=x['width_intent_mismatch_fraction'],L_intent_mismatch=y['width_intent_mismatch_fraction']))
 groups={}
 for cohort in ['all','old_train','old_val','new_val','combined_val']:
  selected=[x for x in rows if cohort=='all' or x['cohort']==cohort or(cohort=='combined_val' and x['cohort'] in ['old_val','new_val'])];gt=[x for x in selected if x['gt_horizon'] is not None];g=dict(window_seed_rows=len(selected),GT_close_switch_rows=len(gt))
  for arm in ['B','L']:
   vals=[abs(x[arm+'_offset']) for x in gt if x[arm+'_offset'] is not None];g[arm]=dict(missing_close_switch_rows=sum(x[arm+'_missing'] for x in gt),abs_offset_distribution=None if not vals else distribution(vals),mean_intent_mismatch=None if not selected else float(np.mean([x[arm+'_intent_mismatch'] for x in selected])))
  groups[cohort]=g
 return dict(note='30Hz target-sequence width-intent proxy only. Source event annotations and predicted closing sequence do NOT identify actual reach-held timing or prove grasp.',groups=groups,rows=rows)

def independence_check():
 s=load(AB/'selection.json');m0=load(s['source_m0_selection']);fresh=load(DATA/'fresh24_heldout.json');aux=load(DATA/'auxiliary_panel.json');manifest=[json.loads(l) for l in Path(s['source_manifest']).read_text().splitlines()];byid={m['episode_id']:m for m in manifest};sets={'AB_train128':s['episodes_train'],'AB_development8':s['development_scenes'],'AB_newval8':s['episodes_new_val'],'AB_heldout12':s['episodes_heldout'],'M0_train_val16':m0['episodes'],'auxiliary24':aux['windows'],'all_filtered_train': [m for m in manifest if m['split']=='train'],'explicit_previously_excluded_val':[byid[i] for i in s['selection_rule']['excluded_previous_val_ids'] if i in byid]};fids={x['episode_id']for x in fresh['scenes']};fseeds={x['seed']for x in fresh['scenes']};assert len(fids)==len(fseeds)==24
 report={}
 for name,rows in sets.items():
  ids={x['episode_id']for x in rows};seeds={x['seed']for x in rows};ids_overlap=sorted(ids&fids);seeds_overlap=sorted(seeds&fseeds);assert not ids_overlap and not seeds_overlap,(name,ids_overlap,seeds_overlap);report[name]=dict(unique_episode_ids=len(ids),unique_scene_seeds=len(seeds),episode_id_overlap=ids_overlap,scene_seed_overlap=seeds_overlap)
 for e in fresh['scenes']:assert sha(e['annotation'])==e['annotation_sha256'] and load(e['annotation'])['seed']==e['seed']
 old,extra=metadata();union=set(map(key,old))|set(map(key,extra));assert len(union)==335
 return dict(status='PASS',fresh24_seed_and_episode_disjoint_from_all_sets=True,checks=report,original_windows=240,aux_windows=144,overlap_windows=49,union_unique_windows=335,no_fresh24_model_results_read=True,curation_limit='Historical reliable observable validation cohort, not a random generated-scene population.',sources={str(p):sha(p) for p in [AB/'selection.json',Path(s['source_m0_selection']),DATA/'fresh24_heldout.json',DATA/'auxiliary_panel.json']})

def collect_readonly():
 """Safe progress collection for root reports; this function makes no writes or GPU imports."""
 locations={'B_old240':AB/'arms/B/evaluations/step_8000','B_extra144':EXPERIMENT/'training/reference_B8000_extra/evaluations/step_8000','L_old240':EXPERIMENT/'training/arms/L/evaluations/step_8000','L_extra144':EXPERIMENT/'training/arms/L/auxiliary/evaluations/step_8000'};result={}
 for name,d in locations.items():
  p=d/'report.json';result[name]=dict(path=str(d),complete_report=p.is_file(),npz_count=len(list(d.glob('*.npz'))) if d.exists() else 0)
  if p.is_file():
   r=load(p);result[name].update(step=r['step'],windows=r['windows'],seeds=r['seeds'],report_sha256=sha(p))
 for name,p in {'L_status':EXPERIMENT/'training/arms/L/status.json','L_summary':EXPERIMENT/'training/arms/L/training_summary.json','B_extra_complete':EXPERIMENT/'training/reference_B8000_extra/completed.json'}.items():
  if p.exists():result[name]=load(p)
 return result

def self_test():
 old,aux=metadata();m=old[0];a=read_arrays(AB/'arms/B/evaluations/step_8000'/filename(m),m);record=dict(meta=m,B=a,L=a);same=summarize([record],lambda m:np.ones(30,dtype=bool),100)
 assert all(same['metrics'][k]['L_minus_B_scene_equal_mean']==0 for k in METRICS)
 altered={**a,'errors':{k:v.astype(np.float64)+.005 for k,v in a['errors'].items()}};delta=summarize([dict(meta=m,B=a,L=altered)],lambda m:m['contact_flags'],100)
 assert all(abs(delta['metrics'][k]['L_minus_B_scene_equal_mean']-.005)<1e-12 for k in METRICS)
 assert summarize([record],lambda m:m['target_phase_by_horizon']=='DROP')['status']=='NO_TARGETS'
 assert np.array_equal(a['seeds'],np.array([42,43,44]));independence=independence_check()
 return dict(status='PASS',real_B_npz_errors_independently_reconstructed=True,identity_pair_zero_delta=True,synthetic_constant_error_delta_exact=True,absent_phase_not_silently_zero=True,seed_pair_order_verified=True,no_torch_or_cuda_import='torch' not in sys.modules,independence=independence)

def analyze(args):
 old,aux=metadata();b0=Path(args.b_original);l0=Path(args.l_original);bx=Path(args.b_aux);lx=Path(args.l_aux);original,p0=load_pair(b0,l0,old);extra,px=load_pair(bx,lx,aux);original_by={key(r['meta']):r for r in original};duplicates=0;union=[{**r,'meta':dict(r['meta'])} for r in original];union_by={key(r['meta']):r for r in union}
 for r in extra:
  if key(r['meta']) in original_by:
   before=original_by[key(r['meta'])];duplicates+=1;union_by[key(r['meta'])]['meta']['auxiliary_landmark']=r['meta']['label']
   for arm in ['B','L']:
    assert set(r[arm]['all_arrays'])==set(before[arm]['all_arrays'])
    for k in r[arm]['all_arrays']:assert np.array_equal(r[arm]['all_arrays'][k],before[arm]['all_arrays'][k]),(arm,key(r['meta']),k)
  else:union.append(r)
 assert duplicates==49 and len(union)==335
 groups={};closures={}
 for panel,records in [('original240',original),('auxiliary144',extra),('union335',union)]:groups.update(metric_groups(records,panel));closures[panel]=closure_summary(records)
 out=ROOT/args.output;assert not out.exists();out.mkdir();report=dict(schema='orchard_B8000_L8000_paired_offline_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),endpoint=8000,comparison='L minus B; negative physical error delta is better',primary_panels_separate=True,union_deduplicated=True,duplicate49_arrays_bitexact_in_both_models=True,units=dict(position_error_m='Euclidean metres',rotation_error_rad='SO3 geodesic radians',width_error_m='absolute total-opening metres'),aggregation='Physical errors before clipping; pair each identical window/horizon/prediction seed. Pooled target summaries and scene-equal means reported separately. Resample scenes, not overlapping targets or noise seeds; CIs descriptive for curated small cohorts, not population claims.',near_horizon_note='near_h1/h5/h10 indicates observation anchored 1/5/10 recorded targets before first PULL reference; not physical reach timing or verified held event.',selection_note='Auxiliary data and fresh24 never select checkpoints; main endpoint fixed at8000.',event_note='first expert PULL[-.30,+.10] proxy by real target timestamp; not anchor phase; no event labels passed to model.',release_note='DROP and DONE separated. Post-DONE source width decreases do not imply failure to release; actual held release and stable bucket occupancy require closed-loop.',provenance={'original':p0,'auxiliary':px,'script_sha256':sha(__file__),'event_index_sha256':sha(DATA/'event_index.json')},data_independence=independence_check(),groups=groups,closing_proxy=closures)
 save(out/'paired_offline.json',report)
 with(out/'summary.csv').open('x',newline='')as f:
  writer=csv.DictWriter(f,fieldnames=['group','metric','status','scenes','windows','paired_seed_target_count','B_pooled_mean','L_pooled_mean','L_minus_B_pooled','B_scene_equal','L_scene_equal','L_minus_B_scene_equal','improved_scenes','worsened_scenes','CI95_low','CI95_high']);writer.writeheader()
  for group,r in groups.items():
   if r['status']!='OK':writer.writerow(dict(group=group,status=r['status'],scenes=0,windows=0,paired_seed_target_count=0));continue
   for metric,v in r['metrics'].items():writer.writerow(dict(group=group,metric=metric,status='OK',scenes=r['scenes'],windows=r['windows'],paired_seed_target_count=r['paired_seed_target_count'],B_pooled_mean=v['B_pooled']['mean'],L_pooled_mean=v['L_pooled']['mean'],L_minus_B_pooled=v['paired_pooled_delta']['mean'],B_scene_equal=v['B_scene_equal_mean'],L_scene_equal=v['L_scene_equal_mean'],L_minus_B_scene_equal=v['L_minus_B_scene_equal_mean'],improved_scenes=v['improved_scenes'],worsened_scenes=v['worsened_scenes'],CI95_low=v['scene_resample_descriptive_CI95'][0],CI95_high=v['scene_resample_descriptive_CI95'][1]))
 lines=['B8000 vs L8000配对离线分析','原240与新增144分别报告；联合335已去重49，两个模型重叠NPZ逐array完全一致。','负的L−B表示误差降低。单位position m、rotation rad、width m。这里只报告预测误差，不构成抓持或释放成功。','']
 for panel,target,band in [('original240','contact_event','first5'),('original240','contact_event','first30'),('auxiliary144','contact_event','first5'),('auxiliary144','DROP','first30'),('auxiliary144','DONE','first30')]:
  r=groups[f'{panel}/combined_val/{target}/{band}'];lines.append(f'{panel} / combined_val / {target} / {band}:')
  if r['status']!='OK':lines.append('  NO_TARGETS');continue
  for metric,v in r['metrics'].items():lines.append(f"  {metric}: scene-equal B={v['B_scene_equal_mean']:.6g}, L={v['L_scene_equal_mean']:.6g}, delta={v['L_minus_B_scene_equal_mean']:+.6g}; better/worse={v['improved_scenes']}/{v['worsened_scenes']} of {r['scenes']}")
 with(out/'report.txt').open('x')as f:f.write('\n'.join(lines)+'\n')
 print(json.dumps(dict(status='COMPLETE',output=str(out),groups=len(groups),original=240,auxiliary=144,union=335)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--inspect',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--b-original',default=str(AB/'arms/B/evaluations/step_8000'));p.add_argument('--l-original',default=str(EXPERIMENT/'training/arms/L/evaluations/step_8000'));p.add_argument('--b-aux',default=str(EXPERIMENT/'training/reference_B8000_extra/evaluations/step_8000'));p.add_argument('--l-aux',default=str(EXPERIMENT/'training/arms/L/auxiliary/evaluations/step_8000'));p.add_argument('--output',default='B8000_vs_L8000');args=p.parse_args()
 if args.preflight:
  r=self_test();save(ROOT/'analysis_preflight.json',r);save(ROOT/'data_independence.json',r['independence']);print(json.dumps(r,indent=2))
 elif args.inspect:print(json.dumps(collect_readonly(),indent=2))
 elif args.run:analyze(args)
 else:p.error('Choose --preflight, --inspect, or --run')
if __name__=='__main__':main()
