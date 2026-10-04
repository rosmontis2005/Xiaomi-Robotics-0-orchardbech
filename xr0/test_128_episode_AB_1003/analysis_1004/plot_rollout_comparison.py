"""Passive CPU plotting of completed fixed-protocol rollouts; never changes control."""
import argparse, hashlib, json, os
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'analysis_1004'/'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def load_run(provider,scene):
    p=ROOT/'closed_loop_1004'/'rollouts'/f'{provider}_reach-conditioned_{scene}'
    c=json.loads((p/'completed.json').read_text())
    for f in ['steps.jsonl','chunks.jsonl','metrics.json']:
        assert hashlib.sha256((p/f).read_bytes()).hexdigest()==c['files_sha256'][f]
    return p,[json.loads(x) for x in (p/'steps.jsonl').read_text().splitlines()], [json.loads(x) for x in (p/'chunks.jsonl').read_text().splitlines()],json.loads((p/'metrics.json').read_text())

def plot(scene):
    runs={label:load_run(provider,scene) for label,provider in [('GT','gt'),('A8000','A_step8000_rng42'),('B8000','B_step8000_rng42')]}
    dest=ROOT/'analysis_1004'/'rollout_figures';dest.mkdir(exist_ok=True)
    fig,axs=plt.subplots(3,3,figsize=(16,10),sharex='col',constrained_layout=True)
    for col,(label,(path,rows,chunks,m)) in enumerate(runs.items()):
        t=np.array([r['control_step'] for r in rows]); cutoff=m['first_grasp_control_step']
        dist=np.array([r['grasp_after'].get('planned_apple',{}).get('tcp_distance_m',np.nan)*100 for r in rows]); dist[t>=cutoff if cutoff is not None else t<0]=np.nan
        targetdist=np.array([np.linalg.norm(np.array(r['target_position'])-r['grasp_after']['planned_apple']['world_position'])*100 if r['grasp_after'].get('planned_apple') else np.nan for r in rows]);targetdist[t>=cutoff if cutoff is not None else t<0]=np.nan
        axs[0,col].plot(t,dist,label='actual TCP to planned fruit');axs[0,col].plot(t,targetdist,alpha=.65,label='command target to planned fruit')
        axs[0,col].set_title(f'{label}: grasp={cutoff}, bucket={m["success"]}')
        axs[1,col].plot(t,[r['target_width']*1000 for r in rows],label='target');axs[1,col].plot(t,[r['measured_width']*1000 for r in rows],label='actual',alpha=.7)
        axs[2,col].plot(t,[r['position_error_m']*100 for r in rows],label='position tracking error')
        tm=[r['control_step'] for r in rows if r['advance_reason']=='maximum_dwell'];axs[2,col].scatter(tm,[0]*len(tm),marker='x',color='red',label='dwell timeout')
        for row in range(3):
            ax=axs[row,col];ax.set_xlim(0,300);ax.grid(alpha=.25)
            if cutoff is not None and cutoff<=300:ax.axvline(cutoff,color='green',linestyle='--',label='first grasp' if row==0 else None)
            for c in chunks[1:]:
                if c['at_control_step']<=300:ax.axvline(c['at_control_step'],color='gray',alpha=.2)
        axs[2,col].set_xlabel('Control step (reach-conditioned, 30 Hz)')
    for row in range(3):
        shown=[]
        for ax in axs[row]:
            for line in ax.lines:
                x=np.asarray(line.get_xdata());y=np.asarray(line.get_ydata())
                if x.size>2:shown.extend(y[(x<=300)&np.isfinite(y)].tolist())
        upper=max(shown)*1.06 if shown else 1
        for ax in axs[row]:ax.set_ylim(0,upper)
    axs[0,0].set_ylabel('Pre-grasp distance (cm)');axs[1,0].set_ylabel('Total gripper opening (mm)');axs[2,0].set_ylabel('Target tracking error (cm)')
    for row in range(3):axs[row,0].legend(fontsize=8)
    fig.suptitle(f'Scene {scene}, inference seed42; columns use independent rollout clocks. Gray lines: replans.\nPlanned fruit is a passive diagnostic only; pre-grasp distances exclude post-grasp samples.')
    fig.savefig(dest/f'scene_{scene}_first300.png',dpi=150);plt.close(fig)
    gt=runs['GT'][2][0];origin=np.array(gt['target_positions'][0]); end=np.array(gt['target_positions'][-1]);e=end[:2]-origin[:2];e/=np.linalg.norm(e)
    fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for label,(_,rows,chunks,m) in runs.items():
        c=chunks[0];points=np.array(c['target_positions']);u=(points[:,:2]-origin[:2])@e*100
        line=axs[0].plot(u,points[:,2]*100,'o-',ms=3,label=label)[0]
        for k in [0,4,14,29]:axs[0].annotate(str(k+1),(u[k],points[k,2]*100),fontsize=8,color=line.get_color())
        axs[1].plot(np.arange(1,31),np.array(c['target_widths'])*1000,label=label)
    axs[0].set_xlabel('Projection along initial GT horizontal approach (cm)');axs[0].set_ylabel('World height (cm)');axs[0].set_aspect('equal',adjustable='datalim')
    axs[1].set_xlabel('Predicted target index');axs[1].set_ylabel('Target total gripper opening (mm)')
    for ax in axs:ax.legend();ax.grid(alpha=.25)
    fig.suptitle(f'Scene {scene}: first chunk, unclipped targets; same reset, inference seed42\n2D projection omits lateral error; these are predicted targets, not measured motion.')
    fig.savefig(dest/f'scene_{scene}_firstchunk.png',dpi=150);plt.close(fig)
    return {'scene':scene,'inputs':{label:str(r[0]) for label,r in runs.items()},'status':'PASS','scope':'passive completed-record plots'}
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--scenes',type=int,nargs='+',required=True);o=a.parse_args()
    for s in o.scenes:
        r=plot(s);(ROOT/'analysis_1004'/'rollout_figures'/f'scene_{s}_plot_validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
