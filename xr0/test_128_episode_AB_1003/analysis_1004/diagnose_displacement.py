"""Auxiliary CPU description of already saved prediction amplitude/direction.
Not a new panel, model selection metric, or inference about visual attention.
"""
import os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.dont_write_bytecode=True
os.environ.update(CUDA_VISIBLE_DEVICES='',MPLCONFIGDIR=str(HERE/'matplotlib_cache'))
import collections,hashlib,json,csv
import numpy as np

EPS=1e-6
DIRECTION_EPS=.001
PANELS=['old_train','old_val','new_val','val_combined']
MODELS=['A0','A8000','B4000','B8000']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(a):
 a=np.asarray([x for x in a if x is not None],dtype=float)
 return None if not len(a) else dict(count=len(a),mean=float(a.mean()),p10=float(np.quantile(a,.1)),median=float(np.median(a)),p90=float(np.quantile(a,.9)),min=float(a.min()),max=float(a.max()))
def ratio(a,b):return None if b<=EPS else float(a/b)
def groups(w):return (['reset'] if w['frame']==0 else [])+['anchor_'+w['anchor_phase']]

def main():
 out=HERE/'displacement_diagnostic';out.mkdir(exist_ok=False)
 selection=json.loads((ROOT/'selection.json').read_text());windows={w['window_id']:w for values in selection['evaluation_sets'].values() for w in values}
 trajectories={};rows=[];sources={}
 for model in MODELS:
  arm=model[0];step=int(model[1:]);path=ROOT/f'arms/{arm}/evaluations/step_{step:04d}/report.json';report=json.loads(path.read_text());sources[str(path.relative_to(ROOT))]=sha(path)
  for row in report['window_metrics']:
   meta=windows[row['window_id']];ap=meta['annotation']
   if ap not in trajectories:trajectories[ap]=json.loads(Path(ap).read_text());assert sha(ap)==meta['annotation_sha256']
   anchor=np.asarray(trajectories[ap]['proprios']['ee_pos'][meta['frame']],dtype=float)
   with np.load(row['npz']) as z:
    gt=z['gt_world_position']-anchor;pred=z['predicted_world_position']-anchor;gn=np.linalg.norm(gt,axis=-1);pn=np.linalg.norm(pred,axis=-1)
    for si,seed in enumerate(z['prediction_seeds']):
     vec=pred[si];r=dict(checkpoint=model,panel=row['evaluation_set'],window_id=row['window_id'],episode_id=row['episode_id'],scene_seed=row['seed'],inference_seed=int(seed),frame=row['frame'],anchor_phase=row['anchor_phase'],groups=groups(row))
     for h in [1,5,15,30]:
      j=h-1;r[f'gt_h{h}_amplitude_m']=float(gn[j]);r[f'pred_h{h}_amplitude_m']=float(pn[si,j]);r[f'h{h}_amplitude_pred_over_gt']=ratio(pn[si,j],gn[j])
      valid=gn[j]>DIRECTION_EPS and pn[si,j]>DIRECTION_EPS
      r[f'h{h}_direction_angle_deg']=float(np.degrees(np.arccos(np.clip(np.dot(gt[j],vec[j])/(gn[j]*pn[si,j]),-1,1)))) if valid else None
      if gn[j]>DIRECTION_EPS:
       projection=np.dot(vec[j],gt[j])/gn[j];r[f'h{h}_signed_longitudinal_error_m']=float(projection-gn[j]);r[f'h{h}_lateral_error_m']=float(np.linalg.norm(vec[j]-projection*gt[j]/gn[j]))
      else:r[f'h{h}_signed_longitudinal_error_m']=None;r[f'h{h}_lateral_error_m']=None
     r.update(gt_h30_over_h1=ratio(gn[-1],gn[0]),pred_h30_over_h1=ratio(pn[si,-1],pn[si,0]),gt_h1_below_1mm=bool(gn[0]<.001),pred_h1_below_1mm=bool(pn[si,0]<.001),gt_first_to_last_m=float(np.linalg.norm(gt[-1]-gt[0])),pred_first_to_last_m=float(np.linalg.norm(vec[-1]-vec[0])),gt_path_length_m=float(np.linalg.norm(np.diff(gt,axis=0),axis=-1).sum()),pred_path_length_m=float(np.linalg.norm(np.diff(vec,axis=0),axis=-1).sum()),gt_within_chunk_rms_spread_m=float(np.sqrt(np.mean(np.sum((gt-gt.mean(0))**2,axis=-1)))),pred_within_chunk_rms_spread_m=float(np.sqrt(np.mean(np.sum((vec-vec.mean(0))**2,axis=-1)))))
     r['first_to_last_pred_over_gt']=ratio(r['pred_first_to_last_m'],r['gt_first_to_last_m'])
     r['path_length_pred_over_gt']=ratio(r['pred_path_length_m'],r['gt_path_length_m'])
     rows.append(r)
 summary={}
 metadata={'checkpoint','panel','window_id','episode_id','scene_seed','inference_seed','frame','anchor_phase','groups','gt_h1_below_1mm','pred_h1_below_1mm'}
 metrics=[k for k in rows[0] if k not in metadata]
 for model in MODELS:
  for panel in PANELS:
   allowed=['old_val','new_val'] if panel=='val_combined' else [panel]
   for group in ['reset','anchor_REACH','anchor_GRASP','anchor_PULL','anchor_TRANSPORT']:
    rr=[r for r in rows if r['checkpoint']==model and r['panel'] in allowed and group in r['groups']]
    if not rr:continue
    eps=sorted(set(r['episode_id'] for r in rr));aggregated={m:stats(r[m] for r in rr) for m in metrics}
    scene={ep:{m:(stats(r[m] for r in rr if r['episode_id']==ep)['mean'] if stats(r[m] for r in rr if r['episode_id']==ep) else None) for m in metrics} for ep in eps}
    angles=[r['h30_direction_angle_deg'] for r in rr if r['h30_direction_angle_deg'] is not None];amps=[r['h30_amplitude_pred_over_gt'] for r in rr if r['h30_amplitude_pred_over_gt'] is not None]
    summary[f'{model}/{panel}/{group}']=dict(windows=len(set(r['window_id'] for r in rr)),episodes=len(eps),window_noise_repetitions=len(rr),pooled_stats=aggregated,scene_mean_stats={m:stats(scene[e][m] for e in eps) for m in metrics},per_scene_mean=scene,descriptive_counts=dict(h30_amplitude_ratio_below_half=sum(x<.5 for x in amps),h30_amplitude_ratio_above_twice=sum(x>2 for x in amps),h30_amplitude_ratio_valid=len(amps),h30_direction_above_30deg=sum(x>30 for x in angles),h30_direction_above_90deg=sum(x>90 for x in angles),h30_direction_valid=len(angles),pred_chunk_rms_spread_below_1mm=sum(r['pred_within_chunk_rms_spread_m']<.001 for r in rr),gt_h1_below_1mm=sum(r['gt_h1_below_1mm'] for r in rr),pred_h1_below_1mm=sum(r['pred_h1_below_1mm'] for r in rr)))
 result=dict(schema='orchard_prediction_displacement_auxiliary_v1',script_sha256=sha(__file__),selection_sha256=sha(ROOT/'selection.json'),auxiliary_only=True,primary_panels_or_metrics_changed=False,model_inference_run=False,heldout_outcomes_read=False,anchor='Expert observation ee_pos at the fixed window frame; world positions in NPZ share that fixed chunk anchor.',ratios='h30/h1 is the ratio of displacement magnitudes relative to the common anchor, NOT ratio of error, and NOT the distance between the two targets.',denominator_rule='Ratios omitted for denominator<=1e-6m; separately count h1<1mm. Directions omitted unless both GT/pred amplitudes>1mm.',nearly_constant_rule='Descriptive count of within-chunk position RMS spread<1mm; not a learned-policy diagnosis threshold.',limits=['Paired trajectory entries share scenes and overlap targets; three generation seeds are not independent scenes.','GT path may curve; endpoint angle/projection describes only the displacement to that horizon, not a full-path direction judgment.','Low endpoint amplitude or low within-chunk spread cannot by itself show visual input is ignored.','Fixed panels contain no DROP anchors/targets; late release remains unmeasured.'],summary=summary,per_window_seed=rows,source_reports_sha256=sources)
 with (out/'displacement.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 with (out/'per_window_seed.csv').open('x',newline='') as f:
  keys=[k for k in rows[0] if k!='groups'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rows)
 lines=['Auxiliary displacement/direction diagnosis; no new inference, no panel/primary metric changes.','All displayed rows use16 fixed validation scenes and pooled window×noise repetitions; full per-scene means saved inJSON.','h30/h1 measures relative-anchor displacement magnitudes; small h1 denominators can inflate this ratio.','', 'checkpoint / anchor group: h30 GT,pred mean cm; endpoint amplitude ratio median; endpoint angle median deg; within-chunk RMS mean cm; predicted h30/h1 median']
 for model in MODELS:
  for group in ['reset','anchor_GRASP','anchor_TRANSPORT']:
   rec=summary[f'{model}/val_combined/{group}'];s=rec['pooled_stats']
   lines.append(f"{model} / {group}: {100*s['gt_h30_amplitude_m']['mean']:.3f}, {100*s['pred_h30_amplitude_m']['mean']:.3f}; {s['h30_amplitude_pred_over_gt']['median']:.3f}; {s['h30_direction_angle_deg']['median']:.2f}; {100*s['pred_within_chunk_rms_spread_m']['mean']:.3f}; {s['pred_h30_over_h1']['median']:.2f}")
   lines.append('  counts '+json.dumps(rec['descriptive_counts']))
 lines.extend(['']+result['limits']);(out/'displacement_report.txt').write_text('\n'.join(lines)+'\n');(out/'diagnose_displacement_source.py').write_text(Path(__file__).read_text())
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
 for ax,group in zip(axes,['reset','anchor_GRASP','anchor_TRANSPORT']):
  for model,color in [('A8000','#d07828'),('B8000','#197991'),('B4000','#93c8d4')]:
   rr=[r for r in rows if r['checkpoint']==model and r['panel'] in ['old_val','new_val'] and group in r['groups']]
   ax.scatter([100*r['gt_h30_amplitude_m'] for r in rr],[100*r['pred_h30_amplitude_m'] for r in rr],label=model,alpha=.45,s=16,color=color)
  maximum=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,maximum],[0,maximum],'k--',lw=1);ax.set_xlim(left=0);ax.set_ylim(bottom=0);ax.set_title(group);ax.set_xlabel('GT anchor→target30 magnitude (cm)');ax.set_ylabel('Predicted magnitude (cm)');ax.legend();ax.grid(alpha=.2)
 fig.suptitle('Saved validation predictions: target30 displacement amplitude; dots are correlated window/noise entries')
 fig.savefig(out/'endpoint_amplitude.png',dpi=170);plt.close(fig)
 print('PASS',out)

if __name__=='__main__':main()
