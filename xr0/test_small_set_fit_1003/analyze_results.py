#!/usr/bin/env python3
"""Independent physical-metric and training-exposure audit for the M0 experiment."""
from pathlib import Path
import json
import sys
sys.dont_write_bytecode=True
import bootstrap
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
GROUPS={'h1':slice(0,1),'h1_5':slice(0,5),'h6_15':slice(5,15),'h16_30':slice(15,30),'full30':slice(0,30)}

def summarize(arrays):
    result={}
    for metric in ['position_error_m','rotation_error_rad','width_error_m']:
        a=np.concatenate([x[metric] for x in arrays],axis=0)
        result[metric]={k:{'mean':float(a[:,v].mean()),'p90':float(np.percentile(a[:,v],90)),'max':float(a[:,v].max())} for k,v in GROUPS.items()}
        result[metric]['by_horizon_mean']=a.mean(axis=0).tolist()
    return result

def main():
    selection=json.loads((R/'selection.json').read_text())
    window_meta={}
    for ep in selection['episodes']:
        for w in ep['windows']:
            window_meta[(ep['split'],ep['episode_id'],int(w['frame']))]={**ep,**w}
    stats=json.loads(Path(selection['stats_path']).read_text())
    mu=np.asarray(stats['mean'])
    experiments={}; baseline_rows=[]
    for directory in sorted((R/'evaluations').glob('step_*')):
        if not (directory/'report.json').exists():
            continue
        rows=[]
        for p in sorted(directory.glob('*.npz')):
            stem=p.stem; split=stem.split('_',1)[0]; eid=stem[len(split)+1:].rsplit('_frame',1)[0]; frame=int(stem.rsplit('_frame',1)[1])
            meta=window_meta[(split,eid,frame)]
            with np.load(p) as z:
                arrays={k:z[k].copy() for k in ['position_error_m','rotation_error_rad','width_error_m']}
                assert all(a.shape==(3,30) and np.isfinite(a).all() for a in arrays.values()), str(p)
                gt=z['gt_action_physical']; pred=z['predicted_action_physical']
                assert np.isfinite(pred).all()
                assert np.allclose(arrays['position_error_m'],np.linalg.norm(pred[:,:,:3]-gt[None,:,:3],axis=-1),atol=1e-5)
                assert np.allclose(arrays['width_error_m'],np.abs(pred[:,:,6]-gt[None,:,6]),atol=1e-7)
                row=dict(split=split,episode_id=eid,frame=frame,anchor_phase=meta.get('anchor_phase',meta.get('phase','unknown')),label=meta.get('label',''),**arrays)
                rows.append(row)
                if directory.name=='step_0000':
                    baseline_rows.append(dict(split=split,episode_id=eid,frame=frame,anchor_phase=row['anchor_phase'],label=row['label'],
                        position_error_m=np.linalg.norm(mu[:,:3]-gt[:,:3],axis=-1)[None],
                        rotation_error_rad=(Rotation.from_rotvec(gt[:,3:6]).inv()*Rotation.from_rotvec(mu[:,3:6])).magnitude()[None],
                        width_error_m=np.abs(mu[:,6]-gt[:,6])[None]))
        assert len(rows)==160, (directory,len(rows))
        result={}
        for split in ['train','val']:
            sr=[x for x in rows if x['split']==split]
            result[split]={'all':summarize(sr),'reset':summarize([x for x in sr if x['frame']==0]),'by_phase':{phase:summarize([x for x in sr if x['anchor_phase']==phase]) for phase in sorted({x['anchor_phase'] for x in sr})}}
        experiments[directory.name]=result
    assert experiments, 'No completed evaluations'
    mean_action={split:{'all':summarize([x for x in baseline_rows if x['split']==split]),'reset':summarize([x for x in baseline_rows if x['split']==split and x['frame']==0])} for split in ['train','val']}
    steps=[json.loads(line) for line in (R/'training_steps.jsonl').read_text().splitlines() if line.strip()] if (R/'training_steps.jsonl').exists() else []
    counts={}; val_steps=[]
    for x in steps:
        eid=x.get('episode_id'); frame=x.get('frame'); key=f'{eid}:{frame}'
        counts[key]=counts.get(key,0)+1
        if (eid,frame) not in {(k[1],k[2]) for k in window_meta if k[0]=='train'}: val_steps.append(x)
    summary={'status':'independent_metric_audit','evaluations':experiments,'mean_action':mean_action,'training_exposure':{'updates':len(steps),'unique_windows':len(counts),'window_counts':counts,'non_train_updates':val_steps},'interpretation':'Errors against one expert demonstration; generated seeds are repetitions, not independent episodes. Mean-action baseline uses original horizon-conditioned physical normalization mean.'}
    (R/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    metric_labels=[('position_error_m','Position error (cm)',100.),('rotation_error_rad','Orientation error (rad)',1.),('width_error_m','Width error (mm)',1000.)]
    fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    xs=[int(k.split('_')[1]) for k in experiments]
    for row, subset in enumerate(['all','reset']):
        for col,(metric,label,scale) in enumerate(metric_labels):
            ax=axs[row,col]
            for split,color in [('train','tab:blue'),('val','tab:orange')]:
                for segment,style in [('h1_5','-'),('full30','--')]:
                    ys=[experiments[k][split][subset][metric][segment]['mean']*scale for k in experiments]
                    ax.plot(xs,ys,style,marker='o',color=color,label=f'{split} {segment}')
            ax.set(title=f'{subset}: {label}',xlabel='Additional optimizer updates',ylabel=label); ax.grid(alpha=.25)
            if row==0 and col==0: ax.legend(fontsize=8)
    fig.savefig(R/'generated_action_learning_curves.png',dpi=150);plt.close(fig)
    if steps:
        losses=np.array([x.get('flow_loss',x.get('loss',np.nan)) for x in steps]); fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
        ax.plot(np.arange(1,len(losses)+1),losses,alpha=.25,label='Per-update flow loss')
        if len(losses)>=50: ax.plot(np.arange(50,len(losses)+1),np.convolve(losses,np.ones(50)/50,mode='valid'),label='50-update mean')
        ax.set(xlabel='Additional optimizer updates',ylabel='Original flow matching loss');ax.grid(alpha=.25);ax.legend();fig.savefig(R/'training_loss.png',dpi=150);plt.close(fig)
    print(json.dumps({'completed_evaluation_steps':xs,'training_updates':len(steps),'unique_train_windows':len(counts),'non_train_updates':len(val_steps),'output':str(R/'analysis.json')}))
if __name__=='__main__': main()
