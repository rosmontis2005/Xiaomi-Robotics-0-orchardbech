#!/usr/bin/env python3
"""Standalone diagnostic figure from hash-verified completed raw trajectories only."""
import sys,os,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;EVAL=ROOT.parent/'evaluation';cache=ROOT/'cache/matplotlib';cache.mkdir(parents=True,exist_ok=True);os.environ['MPLCONFIGDIR']=str(cache)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(arm):
 folder=EVAL/'rollouts'/f'B8000_targets{arm}_rng42_reach-conditioned_2013322';complete=json.loads((folder/'completed.json').read_text());assert complete['status']=='COMPLETE'
 for name in ['steps.jsonl','chunks.jsonl','summary.json']:assert sha(folder/name)==complete['files_sha256'][name]
 rows=[json.loads(l)for l in(folder/'steps.jsonl').read_text().splitlines()];rows=[r for r in rows if r['control_step']<=300];chunks=[json.loads(l)for l in(folder/'chunks.jsonl').read_text().splitlines()];close=next((r['control_step']for r in rows if r['grasp_after']['gripper_intent']<0),None);held=next((r['control_step']for r in rows if r['grasp_after']['held_apple_id'] is not None),None);before=rows[0]['grasp_before']['planned_apple'];appleids={r['grasp_after']['planned_apple']['apple_id']for r in rows};assert len(appleids)==1 and before['apple_id'] in appleids
 return dict(folder=str(folder),steps=rows,chunks=chunks,close=close,held=held,time=np.r_[0,[r['control_step']for r in rows]],xyz=np.array([before['hand_local_position']]+[r['grasp_after']['planned_apple']['hand_local_position']for r in rows])*1000,target_width=np.array([r['target_width']for r in rows])*1000,actual_width=np.array([r['grasp_after']['measured_width_m']for r in rows])*1000,planned_apple_id=before['apple_id'],source_sha256=complete['files_sha256'])

def main():
 data={'B30':read(30),'B5':read(5)};blue='#2166ac';orange='#d95f02';colors={'B30':blue,'B5':orange};fig,axes=plt.subplots(4,1,figsize=(11,11.7),sharex=True,gridspec_kw={'height_ratios':[1.25,1,1.15,1]});fig.subplots_adjust(top=.84,bottom=.105,left=.115,right=.965,hspace=.18)
 fig.suptitle('Early re-planning changes the approach path (scene 2013322)',fontsize=15,y=.977)
 fig.text(.115,.942,'Same B8000 checkpoint, same initial 30 targets, inference seed 42.\nB30: first grasp at step '+str(data['B30']['held'])+'.  B5: no grasp in 900 steps.',fontsize=10,va='top')
 bounds=[(-25,25),(-40,40),(65,125)];names=['Palm x (mm)','Palm y (mm)','Palm z (mm)']
 for i,ax in enumerate(axes[:3]):
  lo,hi=bounds[i];ax.axhspan(lo,hi,color='#5abf90',alpha=.17,zorder=0);ax.axhline(lo,color='#31865d',lw=.7,ls=':');ax.axhline(hi,color='#31865d',lw=.7,ls=':')
  for arm,d in data.items():ax.plot(d['time'],d['xyz'][:,i],color=colors[arm],lw=1.7,label=arm)
  ax.set_ylabel(names[i]);ax.grid(axis='y',alpha=.18)
  ax.text(.012,.04,f'Eligible interval: {lo} to {hi} mm',transform=ax.transAxes,fontsize=9,color='#246143')
 for arm,d in data.items():
  x=[r['control_step']for r in d['steps']];axes[3].plot(x,d['actual_width'],color=colors[arm],lw=1.65,label=arm+' actual');axes[3].plot(x,d['target_width'],color=colors[arm],lw=1.05,ls='--',alpha=.9,label=arm+' predicted target')
 axes[3].set_ylabel('Total opening (mm)');axes[3].grid(axis='y',alpha=.18);axes[3].set_xlabel('Completed control steps (30 Hz; target advancement remains reach-conditioned)');axes[3].legend(loc='lower right',ncol=2,fontsize=9)
 for ax in axes:
  ax.set_xlim(0,300);ax.axvspan(data['B30']['held'],300,color=blue,alpha=.045,zorder=-1)
  for arm,d in data.items():
   if d['close'] is not None:ax.axvline(d['close'],color=colors[arm],ls='--',lw=1.0,alpha=.8)
  ax.axvline(data['B30']['held'],color=blue,ls=':',lw=1.6)
  for c in data['B5']['chunks'][1:4]:ax.axvline(c['at_control_step'],color=orange,ls=':',lw=.65,alpha=.45)
 axes[0].annotate('B30 closing\nstep '+str(data['B30']['close']),xy=(data['B30']['close'],data['B30']['xyz'][data['B30']['close'],0]),xytext=(15,450),arrowprops={'arrowstyle':'->','color':blue},color=blue,fontsize=9)
 axes[0].annotate('B5 closing\nstep '+str(data['B5']['close']),xy=(data['B5']['close'],data['B5']['xyz'][data['B5']['close'],0]),xytext=(122,405),arrowprops={'arrowstyle':'->','color':orange},color=orange,fontsize=9)
 axes[0].text(.50,.1,'B5 first 3 consumed prefixes:\n179 / 198 / 189 mm predicted displacement\n171 / 191 / 180 mm actual displacement',transform=axes[0].transAxes,fontsize=9,bbox={'facecolor':'white','alpha':.83,'edgecolor':'.8'})
 fig.legend(handles=[Line2D([0],[0],color=blue,lw=2,label='B30'),Line2D([0],[0],color=orange,lw=2,label='B5'),Patch(facecolor='#5abf90',alpha=.25,label='Palm eligibility interval'),Line2D([0],[0],color='gray',ls='--',label='First negative intent'),Line2D([0],[0],color=blue,ls=':',label='First B30 held')],loc='upper left',bbox_to_anchor=(.115,.903),ncol=3,fontsize=9,frameon=False)
 fig.text(.115,.065,'Blue background: B30 already holds the fruit; its later palm curve includes post-grasp motion.\nOrange dotted lines: first three B5 re-plans (steps 30, 67, 103). Green bands are necessary geometry limits,\nnot sufficient grasp conditions. All curves are control-boundary observations of the planned fruit.',fontsize=9,va='top')
 fig.text(.115,.016,'Evidence is compatible with repeated coarse approach after re-anchoring; it does not identify a unique learning failure.',fontsize=9,color='.3')
 out=ROOT/'short_commit_seed2013322_v2';assert not out.with_suffix('.png').exists() and not out.with_suffix('.pdf').exists();fig.savefig(out.with_suffix('.png'),dpi=220);fig.savefig(out.with_suffix('.pdf'));plt.close(fig)
 meta=dict(status='COMPLETE',scene_seed=2013322,processed_targets=[30,5],inference_seed=42,planned_apple_ids={k:v['planned_apple_id']for k,v in data.items()},first_negative_intent_steps={k:v['close']for k,v in data.items()},first_held_steps={k:v['held']for k,v in data.items()},source_paths={k:v['folder']for k,v in data.items()},source_hashes={k:v['source_sha256']for k,v in data.items()},figure_files={str(out.with_suffix(s)):sha(out.with_suffix(s))for s in ['.png','.pdf']},script_sha256=sha(__file__),GPU_used=False)
 with(ROOT/'short_commit_seed2013322_v2.json').open('x')as f:json.dump(meta,f,indent=2);f.write('\n')
 print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
