"""Same four policy fields. Recovery metadata and phases are never policy input."""
import bootstrap
import json
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from mibot.data.datasets.orchardbench_dataset import OrchardBenchDataset,policy_messages,load_stats
from mibot.data.datamodule.orchardbench_datamodule import OrchardBenchDataModule
from mibot.utils.io import normalize_action
from treesim.orchard_action import encode_window,prepare_rgb,state_vector,action_mask
from recovery_common import ROOT,ORCHARD,STATS,PROCESSOR

class RecoveryDataset(Dataset):
 def __init__(self,manifest):
  self.rows=[json.loads(l) for l in Path(manifest).read_text().splitlines() if l.strip()];assert all(r['accepted'] for r in self.rows)
  self.samples=[(r['annotation'],frame) for r in self.rows for frame in r['legal_anchors']]
  self.stats,self.mean,self.std=load_stats(STATS);self.episodes={r['annotation']:json.loads(Path(r['annotation']).read_text()) for r in self.rows}
 def __len__(self):return len(self.samples)
 def __getitem__(self,index):
  path,frame=self.samples[index];return self.from_trajectory(self.episodes[path],frame)
 def from_trajectory(self,t,frame):
  assert t['trajectory_type']=='recovery_success' and t['schema']=='orchard_xr0_recovery_R_v1' and t['record_fps']==30
  assert 0<=frame<=t['num_frames']-30  # local declared continuation anchors; no production loader change
  p=t['proprios'];images=[]
  for key in ['ego','wrist_left']:
   reader=VideoReader(t['observations'][key][0]['path'],num_threads=1);assert len(reader)==t['num_frames'];images.append(prepare_rgb(Image.fromarray(reader[frame].asnumpy())))
  return dict(messages=policy_messages(images),action=torch.from_numpy(normalize_action(encode_window(t,frame),self.mean,self.std)),action_mask=torch.from_numpy(action_mask()),state=torch.from_numpy(state_vector(p['ee_pos'][frame],p['ee_rotm'][frame],p['gripper_pos'][frame],p['arm_joint'][frame])))

class ScheduledInputs:
 def __init__(self,schedule,limit=256):
  self.original=OrchardBenchDataset(dict(train_datasets=dict(root=str(ORCHARD/'data/orchard_v1_2650/filtered'),split='train',episode_ids=None,stats_path=str(STATS),action_length=30)))
  self.recovery=RecoveryDataset(ROOT/'recovery_manifest.jsonl');self.schedule=schedule
  self.lookup_B={(str(Path(p).resolve()),f):i for i,(p,f) in enumerate(self.original.samples)};self.lookup_R={(str(Path(p).resolve()),f):i for i,(p,f) in enumerate(self.recovery.samples)}
  self.dm=OrchardBenchDataModule(dict(processor_path=str(PROCESSOR)));self.cache=OrderedDict();self.limit=limit
 def sample(self,row):
  key=(str(Path(row['annotation']).resolve()),row['frame'])
  return self.original[self.lookup_B[key]] if row['source']=='original_B' else self.recovery[self.lookup_R[key]]
 def get(self,cursor):
  row=self.schedule[cursor];key=row['window_id']
  if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
  batch=self.dm.collate_fn([self.sample(row)])
  assert batch['action'].shape==(1,30,32) and batch['action'].dtype==torch.float32
  assert all(isinstance(v,torch.Tensor) and v.device.type=='cpu' for v in batch.values())
  self.cache[key]={k:v.contiguous() for k,v in batch.items()}
  if len(self.cache)>self.limit:self.cache.popitem(last=False)
  return self.cache[key]
