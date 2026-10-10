from common import *
import numpy as np
from PIL import Image,ImageDraw
from treesim.vla_env import OrchardVLAEnv,VLAEnvConfig
from treesim.target_visibility import TargetVisibility
from treesim.orchard_action import prepare_rgb
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(ORCHARD/'experiments/initialization_prior_1009'))
from scenes import select

def inspect(seed):
 env=OrchardVLAEnv(VLAEnvConfig(max_control_steps=2100))
 try:
  obs,info=env.reset(seed=seed);initial=json.loads((DATA/f'seed_{seed}/initial.json').read_text());target=initial['truth']['target_id']
  assert target==env._reset_stance['apple_index']
  v=TargetVisibility(env.tm.model,env.tm.apple_bodies[target],env.static_camera,env.wrist_camera);row=v.check(env.sim.state_0,'initial')
  row['seed']=seed;row['target_position_matches']=bool(np.allclose(env.sim.body_q_np()[env.tm.apple_bodies[target],:3],initial['truth']['fruit_position'],atol=1e-5))
  assert row['target_position_matches']
  for name,mask in v.masks.items():
   image=Image.fromarray(mask.astype('uint8')*255);w,h=image.size;cw,ch=int(w*.95),int(h*.95);x,y=(w-cw)//2,(h-ch)//2;crop=image.crop((x,y,x+cw,y+ch));effective=prepare_rgb(image)
   row[name+'_crop_pixels']=int((np.asarray(crop)>0).sum());row[name+'_resized_mask_ge_half']=int((np.asarray(effective)>=128).sum());row[name+'_resized_mask_nonzero']=int((np.asarray(effective)>0).sum());row[name+'_edge_pixels']=int(mask.sum()-(np.asarray(crop)>0).sum())
  if seed in json.loads((ROOT/'visibility_selection.json').read_text())[:12]:v.save_gallery(ROOT/'visibility_gallery'/str(seed))
  return row
 finally:env.close()

def run():
 if not (ROOT/'scenes.json').exists():write(ROOT/'scenes.json',select(8500000,16))
 seeds=json.loads((ROOT/'visibility_selection.json').read_text());path=ROOT/'visibility.jsonl';done={r['seed'] for r in lines(path)} if path.exists() else set()
 from concurrent.futures import ProcessPoolExecutor
 import multiprocessing
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  for row in pool.map(inspect,[seed for seed in seeds if seed not in done]):
   with path.open('a') as f:f.write(json.dumps(row)+'\n')
   done.add(row['seed']);print('VISIBILITY',len(done),row['seed'],flush=True)
 # Saved RGB phase sheets; decoded current frames only, no rerunning trajectories.
 import subprocess
 for seed in seeds[::15]:
  folder=DATA/f'seed_{seed}';t=json.loads((folder/'trajectory.json').read_text());phases=[c['phase'] for c in t['commands']];frames=[]
  for phase in PHASES:
   idx=[i for i,p in enumerate(phases) if p==phase]
   if idx:frames.append((phase,idx[len(idx)//2] if phase!='GRASP' else max(idx[0],idx[-1]-15)))
  sheet=Image.new('RGB',(192*len(frames),2*164),'white');draw=ImageDraw.Draw(sheet)
  for j,key in enumerate(['rgb_static','rgb_wrist']):
   path_video=folder/t['rgb'][key]
   for k,(phase,f) in enumerate(frames):
    raw=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(path_video),'-vf',f'select=eq(n\\,{f})','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'],capture_output=True,check=True).stdout
    sheet.paste(Image.frombytes('RGB',(192,144),raw),(k*192,j*164+20));draw.text((k*192,j*164),f'{seed} {phase} f{f}',fill='black')
  (ROOT/'contact_sheets').mkdir(exist_ok=True);sheet.save(ROOT/'contact_sheets'/f'{seed}.png')
 rows=lines(path)
 def bucket(n):return '0' if n==0 else '1-16' if n<=16 else '17-64' if n<=64 else '65-128' if n<=128 else '>128'
 from collections import Counter
 stats={key:dict(Counter(bucket(r[key]) for r in rows)) for key in rows[0] if key.endswith(('visible_pixels','crop_pixels','resized_mask_ge_half'))}
 write(ROOT/'visibility_summary.json',dict(n=len(rows),buckets=stats,raw_both_zero=sum(not r['any_policy_view_visible'] for r in rows),limitations='Reset rerender verified target identity/geometry; resized binary-mask coverage is geometric diagnostic, not recognition probability. Phase sheets use saved videos; no exact dynamic target masks.'))
 with (ROOT/'audit_report.md').open('a') as f:f.write('\n## Reset 视觉审计\n\n150 个高度分层确定性样本，原始/95% crop/resize 掩膜分桶见 visibility_summary.json；shape-index 机制复用 TargetVisibility。resize 数值为掩膜覆盖诊断，不是可识别概率。动态阶段只抽样保存视频，无精确逐阶段目标像素计数；不能定量声称 GRASP 前 wrist 通常更好。目标选择来自真值规划器，prompt 未标目标身份；即使可见，也不能据此证明多果实间目标可区分或选择由输入可解释。GRASP 接触、稳定持果与释放速度的单帧可观察性仍有限。没有修改样本或阻止训练。\n')
if __name__=='__main__':run()
