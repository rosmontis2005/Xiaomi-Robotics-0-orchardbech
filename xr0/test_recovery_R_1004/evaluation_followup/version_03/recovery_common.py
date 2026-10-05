import bootstrap
import hashlib, json, importlib.util, time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT,XR0,ORCHARD=bootstrap.ROOT,bootstrap.XR0,bootstrap.ORCHARD
AB=XR0/'test_128_episode_AB_1003'; STATS=ORCHARD/'data/orchard_v1_2650/filtered/action_stats.json';PROCESSOR=XR0.parent/'checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D'
PROTOCOL=json.loads((ROOT/'protocol.json').read_text());BASE=Path(PROTOCOL['baseline']['base_checkpoint']);ENDPOINT=Path(PROTOCOL['baseline']['checkpoint'])
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,Path):return str(x)
 raise TypeError(type(x).__name__)
def write(path,value):
 path=Path(path);assert path.resolve().is_relative_to(ROOT);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,default=serial,allow_nan=False,indent=2)+'\n');tmp.replace(path)
def append(path,value):
 with Path(path).open('a') as f:f.write(json.dumps(value,default=serial,allow_nan=False)+'\n');f.flush()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
strict_mod=module(XR0/'test_grasp_1004/evaluation/strict_metrics.py','r_strict')
StrictPlacement,fruit_snapshot=strict_mod.StrictPlacement,strict_mod.fruit_snapshot

def environment():
 from treesim.vla_env import OrchardVLAEnv,VLAEnvConfig
 return strict_mod.make_continuing_env(OrchardVLAEnv)(VLAEnvConfig(max_control_steps=900,detach_force_scale=1.5,grasp_mode='benchmark_assist'))
def advance(obs,p,r,k,dwell):
 pe=float(np.linalg.norm(p[k]-obs['tcp_pos_world']));re=float((Rotation.from_matrix(r[k])*Rotation.from_quat(obs['tcp_quat_world']).inv()).magnitude())
 return pe<=.01 and re<=.08 or dwell>=30,pe,re

def truth(env):
 # Teacher/debug ONLY. Whitelisted loader never consumes this object.
 bq=env.sim.body_q_np();fruit=int(env._reset_stance['apple_index']);body=int(env.tm.apple_bodies[fruit]);cp,cr=env._chassis_pose()
 from treesim import robot
 bucket=cp+cr.apply([robot._BUCKET_CENTER_X,0.,robot._CHASSIS_Z+robot._CHASSIS[2]+robot._BUCKET_WALL_H])
 helds=np.flatnonzero(env.sim.apples._held_host[:env.sim.apples.n]).tolist()
 return dict(debug_usage='NOT POLICY INPUT',sim_time=env.sim.sim_time,joint_pos=env._obs['joint_pos'],joint_vel=env._obs['joint_vel'],tcp_position=env._obs['tcp_pos_world'],tcp_quaternion=env._obs['tcp_quat_world'],width=env._obs['gripper_width'],fruit_id=fruit,fruit_pose=bq[body],fruit_velocity=env.sim.state_0.body_qd.numpy()[body],bucket_position=bucket,base_pose=bq[env.chassis],held=helds,detached=env.sim.apples.detached.copy(),branch_breaks=env.sim.breaker.broken_count,fruit_bucket_distance=float(np.linalg.norm(bq[body,:3]-bucket)),tcp_fruit_distance=float(np.linalg.norm(env._obs['tcp_pos_world']-bq[body,:3])),palm_local_fruit=Rotation.from_quat(bq[env.wrist,3:]).inv().apply(bq[body,:3]-bq[env.wrist,:3]))
def physics_digest(env):
 h=hashlib.sha256()
 for array in [env.sim.state_0.body_q,env.sim.state_0.body_qd,env.sim.state_0.joint_q,env.sim.state_0.joint_qd]:h.update(array.numpy().tobytes())
 for a in [env.sim.apples.detached,env.sim.apples._held_host,env.sim.apples._hold_body_host]:h.update(np.asarray(a).tobytes())
 h.update(str(env.sim.sim_time).encode());return h.hexdigest()
def update_strict(env,strict,step,held=None):
 fs=fruit_snapshot(env)
 if held is None:held=fs['held_apple_id']
 return strict.update(step,held,fs['detached_apple_ids'],fs['in_bucket_apple_ids'],bool(fs['in_bucket_apple_ids']))
def protection_check():
 assert sha(STATS)==json.loads((AB/'selection.json').read_text())['stats_sha256'],'Original action statistics changed'
 assert sha(ENDPOINT)==PROTOCOL['baseline']['sha256']
 for p,h in PROTOCOL['production_sha256'].items():assert sha(p)==h,p

def make_policy():
 import torch
 from mibot.server.orchard_policy import OrchardPolicy
 from checkpoint_io import load_trainable_overlay
 torch.set_num_threads(1)
 policy=OrchardPolicy(str(BASE),str(PROCESSOR),str(STATS))
 report=load_trainable_overlay(policy.model,ENDPOINT,base_checkpoint=BASE)
 assert report['step']==8000 and report['checkpoint_sha256']==PROTOCOL['baseline']['sha256']
 assert all(p.dtype==torch.float32 for n,p in policy.model.named_parameters() if not n.startswith('vlm.'))
 assert policy.model.num_steps==5
 write(ROOT/'student_load_manifest.json',dict(overlay=report,base=policy.load_report));return policy
