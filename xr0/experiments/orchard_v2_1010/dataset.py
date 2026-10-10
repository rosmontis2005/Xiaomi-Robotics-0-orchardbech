from common import *
from functools import lru_cache
import numpy as np
import torch
from PIL import Image
from decord import VideoReader
from treesim.orchard_action import prepare_rgb,action_mask
from treesim.orchard_command import CONTRACT
from mibot.data.datasets.orchardbench_dataset import policy_messages
from mibot.data.datamodule.orchardbench_datamodule import OrchardBenchDataModule

@lru_cache(maxsize=64)
def arrays(seed):
 with np.load(ROOT/'arrays'/f'{seed}.npz') as z:return {k:z[k] for k in z.files}
@lru_cache(maxsize=32)
def video(path):return VideoReader(str(path),num_threads=1)

class V2Dataset(torch.utils.data.Dataset):
 def __init__(self,rows,split=None):
  self.rows=rows;self.manifest={r['seed']:r for s in ['train','validation'] for r in lines(ROOT/f'{s}_manifest.jsonl')}
  stats=json.loads((ROOT/'action_stats.json').read_text());assert stats['contract']==CONTRACT and stats['source_split']=='train'
  assert stats['train_manifest_sha256']==sha(ROOT/'train_manifest.jsonl')
  self.mean=np.array(stats['mean'],np.float32);self.std=np.array(stats['std'],np.float32)
  for r in rows:
   assert r['seed'] in self.manifest and r['frame']%5==0
   if split:assert self.manifest[r['seed']]['split']==split
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  row=self.rows[i];a=arrays(row['seed']);idx=row['index'];assert a['starts'][idx]==row['frame']
  images=[prepare_rgb(Image.fromarray(video(self.manifest[row['seed']]['video_sources'][k])[row['frame']].asnumpy())) for k in ['rgb_static','rgb_wrist']]
  action=np.zeros((30,32),np.float32);action[:,:7]=a['actions'][idx];action=(action-self.mean)/(self.std+1e-6)
  return dict(messages=policy_messages(images),state=torch.from_numpy(a['state'][idx].copy()),action=torch.from_numpy(action),action_mask=torch.from_numpy(action_mask()))
def collator():return OrchardBenchDataModule(dict(processor_path=str(BASE))).collate_fn
