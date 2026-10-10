from common import *
from collections import Counter,defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import profile

def run():
 rows=lines(ROOT/'training_steps.jsonl') if (ROOT/'training_steps.jsonl').exists() else []
 if rows:
  assert [r['step'] for r in rows]==list(range(1,len(rows)+1));write(ROOT/'actual_exposure.json',profile(rows))
  fig,ax=plt.subplots(figsize=(11,4));x=[r['step'] for r in rows]
  for n in [50,200,1000]:ax.plot(x,[r['rolling'][str(n)] for r in rows],label=f'rolling {n}',lw=.8)
  ax.set(xlabel='actual optimizer updates',ylabel='original flow loss');ax.legend();fig.tight_layout();fig.savefig(ROOT/'training_loss.png',dpi=150);plt.close(fig)
  norms=np.array([r['gradient_norm_before_clip'] for r in rows]);write(ROOT/'optimization_profile.json',dict(updates=len(rows),updates_per_second=len(rows)/sum(r['batch_loading_seconds']+r['optimization_seconds'] for r in rows),data_time_fraction=sum(r['batch_loading_seconds'] for r in rows)/sum(r['batch_loading_seconds']+r['optimization_seconds'] for r in rows),clip_fraction=float(np.mean(norms>1)),gradient_norm_quantiles={str(q):float(np.quantile(norms,q)) for q in [0,.25,.5,.75,.95,.99,1]},peak_allocated=max(r['peak_allocated'] for r in rows),anomalies=[str(p) for p in ROOT.glob('anomaly_*.json')]))
 evals=[json.loads(p.read_text()) for p in sorted((ROOT/'evaluations').glob('step_*.json'))]
 if evals:
  fig,axes=plt.subplots(1,2,figsize=(11,4));x=[e['step'] for e in evals]
  for split in ['train','validation']:axes[0].plot(x,[e['flow_loss'][split] for e in evals],marker='.',label='fixed '+split)
  axes[0].legend();axes[0].set(xlabel='optimizer updates',ylabel='fixed flow loss');axes[1].plot(x,[e['gap'] for e in evals]);axes[1].set(xlabel='optimizer updates',ylabel='validation - train');fig.tight_layout();fig.savefig(ROOT/'fixed_loss_and_gap.png',dpi=150);plt.close(fig)
  offline=[]
  for e in evals:
   for split in ['train','validation']:
    for phase in PHASES+['reset']:
     rr=[r for r in e['records'] if r['split']==split and ((r['frame']==0) if phase=='reset' else r['anchor_phase']==phase) and 'errors' in r]
     if rr:
      for h in ['h1:5','h6:15','h16:30','full30']:offline.append(dict(step=e['step'],split=split,anchor_phase=phase,horizon=h,n=len(rr),**{k:float(np.mean([r['errors'][h][k] for r in rr])) for k in ['position_mm','rotation_rad','width_mm']}))
  write(ROOT/'offline_error_summary.json',offline)
  import csv
  if offline:
   with (ROOT/'offline_error_summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(offline[0]));w.writeheader();w.writerows(offline)
 results=[json.loads(p.read_text()) for p in sorted((ROOT/'closed_loop').glob('*/*/result.json'))];write(ROOT/'closed_loop_cases.json',results)
 aggregate=[]
 for label in ['expert','step_000000','step_010000','step_020000','step_040000','step_060000']:
  for init in ['G0','G-','G+']:
   rr=[r for r in results if r['label']==label and r['initialization']==init]
   if rr:aggregate.append(dict(label=label,init=init,n=len(rr),**{k:sum(bool(r[k]) for r in rr) for k in ['strict_success','legacy_success','entered_grasp','held15','detach','actual_release','reasonable_release','stable_bucket']},valid=sum(r['initialization_valid'] for r in rr),common_valid=sum(r['common_valid'] for r in rr)))
 write(ROOT/'closed_loop_summary.json',aggregate)
 complete=len(rows)==60000 and len(results)==288 and len(evals)==32
 exposure=profile(rows) if rows else {}
 report=['# OrchardBench V2 首轮实验',f'\n状态：{"训练与全部闭环完成" if complete else "尚未完成；以下仅为当前已完成证据"}。实际 optimizer updates：{len(rows)} / 60000。',f'实际独立 episodes/windows：{exposure.get("unique_episodes",0)}/{exposure.get("unique_windows",0)}。', '\n## 固定 train / validation loss', '| step | train | validation | gap |','|---:|---:|---:|---:|']
 for e in evals:report.append(f"| {e['step']} | {e['flow_loss']['train']:.6f} | {e['flow_loss']['validation']:.6f} | {e['gap']:.6f} |")
 report+=['\n## 逐 checkpoint 闭环','| model | geometry | n | strict | held15 | detach | release | reasonable release | stable | legacy |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in aggregate:report.append(f"| {r['label']} | {r['init']} | {r['n']} | {r['strict_success']} | {r['held15']} | {r['detach']} | {r['actual_release']} | {r['reasonable_release']} | {r['stable_bucket']} | {r['legacy_success']} |")
 report+=['\n## 解释边界','训练 loss、固定 flow loss、专家状态动作误差与学生闭环任务成功分别报告。少量配对场景的退化不能单独证明过拟合。G± 是超出本训练横向覆盖的 OOD 诊断；没有无图像对照，不能据成功率断言视觉使用。','V2 使用 requested command、每5个真实边界重规划；模型输出仍是30。与旧 V1 的数据、预算和执行机制不同，不作单因素成功率因果归因。','所有原始逐步日志和权重留在本地。冻结来源见 frozen.json；checkpoint SHA 见 training_complete.json；逐场景与释放原链见 closed_loop_cases.json；采样曝光见 actual_exposure.json。','\n完整研究解释需要在所有闭环完成后人工综合本报告、分阶段离线误差与配对场景证据。本脚本只汇总已存在记录，不生成未经支持的结论。']
 (ROOT/'final_report.md').write_text('\n'.join(report)+'\n');write(ROOT/'analysis_status.json',dict(complete=complete,updates=len(rows),rollouts=len(results),panel_evaluations=len(evals)))
if __name__=='__main__':run()
