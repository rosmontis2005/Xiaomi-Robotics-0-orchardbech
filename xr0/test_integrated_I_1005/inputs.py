"""Current two RGB/current proprio only; metadata never enters policy tensors."""
import bootstrap
import json
from pathlib import Path
from collections import OrderedDict
import torch
from decord import VideoReader
from PIL import Image
from mibot.data.datasets.orchardbench_dataset import policy_messages,load_stats
from mibot.data.datamodule.orchardbench_datamodule import OrchardBenchDataModule
from mibot.utils.io import normalize_action
from treesim.orchard_action import encode_window,prepare_rgb,state_vector,action_mask
from recovery_common import ROOT,ORCHARD,STATS,PROCESSOR

class ScheduledInputs:
    def __init__(self,rows,limit=256):
        self.rows=rows;self.episodes={};self.videos=OrderedDict();self.cache=OrderedDict();self.limit=limit
        _,self.mean,self.std=load_stats(STATS);self.dm=OrchardBenchDataModule(dict(processor_path=str(PROCESSOR)))
    def sample(self,row):
        path=row['annotation']
        if path not in self.episodes:self.episodes[path]=json.loads(Path(path).read_text())
        t=self.episodes[path];frame=row['frame'];p=t['proprios'];images=[]
        assert t['trajectory_type'] in ['success','recovery_success'] and 0<=frame<=t['num_frames']-30
        for key in ['ego','wrist_left']:
            video=Path(t['observations'][key][0]['path'])
            if not video.is_absolute():video=ORCHARD/'data/orchard_v1_2650/filtered'/video
            if str(video) not in self.videos:
                self.videos[str(video)]=VideoReader(str(video),num_threads=1)
                assert len(self.videos[str(video)])==t['num_frames']
            self.videos.move_to_end(str(video));reader=self.videos[str(video)]
            images.append(prepare_rgb(Image.fromarray(reader[frame].asnumpy())))
            if len(self.videos)>16:self.videos.popitem(last=False)
        return dict(messages=policy_messages(images),action=torch.from_numpy(normalize_action(encode_window(t,frame),self.mean,self.std)),action_mask=torch.from_numpy(action_mask()),state=torch.from_numpy(state_vector(p['ee_pos'][frame],p['ee_rotm'][frame],p['gripper_pos'][frame],p['arm_joint'][frame])))
    def get(self,cursor):
        row=self.rows[cursor];key=row['window_id']
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        batch=self.dm.collate_fn([self.sample(row)])
        assert batch['state'].shape==(1,1,32) and batch['action'].shape==(1,30,32)
        self.cache[key]={k:v.contiguous() for k,v in batch.items()}
        if len(self.cache)>self.limit:self.cache.popitem(last=False)
        return self.cache[key]
