"""Four measured proprio tokens at control-time offsets; no past RGB or truth inputs."""
import bootstrap
import json
from pathlib import Path
from collections import deque
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from treesim.orchard_action import state_vector,OrchardActionAdapter
from recovery_dataset import ScheduledInputs as RInputs
from recovery_common import ROOT,XR0
OFFSETS=(-15,-5,-1,0)
MODES=('H0','Hrepeat','Hhistory','Hanchor')

def proprio_at(t,index):
 p=t['proprios']
 return state_vector(p['ee_pos'][index],p['ee_rotm'][index],p['gripper_pos'][index],p['arm_joint'][index])[0]

def measured_from_record(record):
 # Historical recorder put measured robot readings inside its debug container.
 # Extract ONLY these four sensor fields; never return any other metadata.
 return state_vector(record['tcp_position'],Rotation.from_quat(record['tcp_quaternion']).as_matrix(),record['width'],record['joint_pos'][:7])[0]

class CausalHistory:
 def __init__(self):self.trajectories={};self.prefixes={};self.prefix_audit={}
 def trajectory(self,path):
  path=str(path)
  if path not in self.trajectories:self.trajectories[path]=json.loads(Path(path).read_text())
  return self.trajectories[path]
 def prefix(self,t):
  r=t['recovery'];key=(r['source_episode_id'],r['student_inference_seed'])
  folder=XR0/'test_recovery_R_1004/student_sources'/f'{key[0]}_rng{key[1]}'
  if key not in self.prefixes:
   initial=json.loads((folder/'initial.json').read_text());rows=json.loads((folder/'steps.json').read_text())
   assert [x['step'] for x in rows]==list(range(1,len(rows)+1))
   records=[initial['truth']]+[x['truth'] for x in rows]
   assert np.allclose(np.diff([x['sim_time'] for x in records]),1/30,atol=1e-9,rtol=0)
   self.prefixes[key]=np.stack([measured_from_record(x) for x in records])
  prefix=self.prefixes[key];takeover=int(r['student_control_step']);assert takeover<len(prefix)
  assert abs(t['orchardbench']['timestamps'][0]-takeover/30)<1e-9
  # Original measured student trace and regenerated takeover differ within the
  # already accepted prefix gate. Preserve measured samples; do not interpolate.
  original=prefix[takeover];current=proprio_at(t,0)
  delta=np.abs(original-current);assert np.max(delta[:3])<=1e-4 and np.max(delta[6:14])<=1e-4
  rot=Rotation.from_euler('xyz',original[3:6]).inv()*Rotation.from_euler('xyz',current[3:6]);assert rot.magnitude()<5e-4
  self.prefix_audit[t['episode_id']]=dict(takeover_control_step=takeover,position_max_m=float(delta[:3].max()),width_m=float(delta[6]),joint_max_rad=float(delta[7:14].max()),orientation_difference_rad=float(rot.magnitude()),source_folder=str(folder))
  return prefix,takeover
 def states(self,path,frame,mode):
  assert mode in MODES;t=self.trajectory(path);assert t['record_fps']==30
  times=t['orchardbench']['timestamps'];assert np.allclose(np.diff(times),1/30,atol=1e-9,rtol=0)
  current=proprio_at(t,frame)
  if mode=='H0':return current[None]
  if mode=='Hrepeat':return np.repeat(current[None],4,axis=0)
  if 'recovery' in t:
   prefix,takeover=self.prefix(t)
   def at(local):return proprio_at(t,local) if local>=0 else prefix[max(0,takeover+local)]
   reset=prefix[0]
  else:
   def at(local):return proprio_at(t,max(0,local))
   reset=proprio_at(t,0)
  values=[at(frame+d) for d in OFFSETS]
  if mode=='Hanchor':values[0]=reset
  result=np.stack(values).astype(np.float32);assert result.shape==(4,32) and np.isfinite(result).all() and not result[:,14:].any()
  return result

class HistoryInputs(RInputs):
 def __init__(self,schedule,mode,limit=256):
  self.mode=mode;self.history=CausalHistory();super().__init__(schedule,limit)
 def sample(self,row):
  sample=super().sample(row)
  sample['state']=torch.from_numpy(self.history.states(row['annotation'],row['frame'],self.mode))
  return sample

class HistoryBuffer:
 def __init__(self,mode):assert mode in MODES;self.mode=mode;self.clear()
 def clear(self):self.values=deque(maxlen=16);self.reset_state=None;self.last_step=None;self.updates=0
 def observe(self,obs,step):
  step=int(step)
  if self.last_step is not None:assert step==self.last_step+1,'History must update every actual control step'
  else:assert step==0
  state=OrchardActionAdapter.state(obs)[0].copy()
  if self.reset_state is None:self.reset_state=state.copy()
  self.values.append(state);self.last_step=step;self.updates+=1
 def state(self):
  assert self.values;current=self.values[-1]
  if self.mode=='H0':return current[None].copy()
  if self.mode=='Hrepeat':return np.repeat(current[None],4,axis=0)
  values=[self.values[max(0,len(self.values)-1+d)] for d in OFFSETS]
  if self.mode=='Hanchor':values[0]=self.reset_state
  return np.stack(values).astype(np.float32)

from mibot.server.orchard_policy import OrchardPolicy
class HistoryPolicy(OrchardPolicy):
 def __init__(self,*args,mode='H0',**kwargs):
  super().__init__(*args,**kwargs);self.history=HistoryBuffer(mode);self.history_calls=[];original_collate=self.collate
  def collate(samples):
   assert len(samples)==1;samples[0]['state']=torch.from_numpy(self.history.state());return original_collate(samples)
  self.collate_late=original_collate;self.collate=collate
 def reset_history(self,obs):self.history.clear();self.history_calls=[];self.history.observe(obs,0)
 def observe_control_step(self,obs,step):self.history.observe(obs,step)
 def predict(self,obs,seed=42):
  current=OrchardActionAdapter.state(obs)[0]
  assert np.array_equal(current,self.history.values[-1])
  self.history_calls.append(dict(control_step=self.history.last_step,buffer_updates=self.history.updates,token_control_steps=([self.history.last_step] if self.history.mode=='H0' else [self.history.last_step]*4 if self.history.mode=='Hrepeat' else [0,max(0,self.history.last_step-5),max(0,self.history.last_step-1),self.history.last_step] if self.history.mode=='Hanchor' else [max(0,self.history.last_step+d) for d in OFFSETS])))
  return super().predict(obs,seed=seed)
