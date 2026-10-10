from common import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

def summary(x):
 x=np.asarray(x);return dict(mean=x.mean(0).tolist(),std=x.std(0).tolist(),quantiles={str(q):np.quantile(x,q,axis=0).tolist() for q in [0,.01,.05,.25,.5,.75,.95,.99,1]})
def main():
 rows=json.loads((ROOT/'geometry.json').read_text());out={}
 for accepted,name in [(True,'accepted'),(False,'valid_reset_rejected')]:
  rr=[r for r in rows if r['accepted']==accepted];xyz=np.array([r['target_base'] for r in rr]);d=[r['tcp_target_distance'] for r in rr]
  out[name]=dict(n=len(rr),xyz=summary(xyz),distance=summary(d),direction=summary([r['tcp_target_direction_base'] for r in rr]),xz_correlation=float(np.corrcoef(xyz[:,0],xyz[:,2])[0,1]),height_to_x=np.polyfit(xyz[:,2],xyz[:,0],1).tolist())
  plt.figure(1,figsize=(12,3))
  for k in range(3):plt.subplot(1,3,k+1);plt.hist(xyz[:,k],bins=35,density=True,alpha=.5,label=name);plt.xlabel('xyz'[k]+' (m)');plt.legend(fontsize=7)
  plt.figure(2);plt.scatter(xyz[:,0],xyz[:,2],s=5,alpha=.4,label=name);plt.xlabel('base x (m)');plt.ylabel('base z (m)');plt.legend()
  plt.figure(3);plt.hist(d,bins=40,density=True,alpha=.5,label=name);plt.xlabel('initial TCP to fruit (m)');plt.legend()
  if accepted:
   fig=plt.figure(4);ax=fig.add_subplot(projection='3d');ax.scatter(*xyz.T,s=3);ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
 for i,name in enumerate(['xyz_distribution','xz_distribution','tcp_distance','xyz_3d'],1):plt.figure(i);plt.tight_layout();plt.savefig(ROOT/(name+'.png'),dpi=150);plt.close(i)
 write(ROOT/'geometry_stats.json',out)
 a=out['accepted'];b=out['valid_reset_rejected']
 text=f'''# V2 初始空间与可观察性简审\n\n原始 accepted 数据重新计算：{a['n']} 条，有效 reset 未接受 {b['n']} 条。\n\nbase xyz mean = {a['xyz']['mean']} m；std = {a['xyz']['std']} m。x–z Pearson r = {a['xz_correlation']:.6f}。初始 TCP–目标距离 mean/std = {a['distance']['mean']:.5f}/{a['distance']['std']:.5f} m。分位数、范围、方向与拒收对照见 geometry_stats.json；完整逐轨迹见 geometry.json。\n\n原站位带来的横向近零先验仍然存在。x 与高度相关，目标落在狭窄空间带；不能把这里的变化范围解释为任意目标三维定位覆盖。接受/拒收分布为观察性比较，不能单独识别筛选因果。\n\n图：xyz_distribution.png、xyz_3d.png、xz_distribution.png、tcp_distance.png。\n\n视觉审计尚在进行；本文件会在精确 reset 渲染和视频抽样完成后更新。没有据视觉诊断重筛数据。\n'''
 (ROOT/'audit_report.md').write_text(text)
if __name__=='__main__':main()
