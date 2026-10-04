import sys
from pathlib import Path
root=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root))
import bootstrap
import json
import numpy as np
from PIL import Image
from decord import VideoReader
from treesim.orchard_action import prepare_rgb, state_vector, action_mask
from mibot.data.datasets.orchardbench_dataset import policy_messages
from mibot.data.datamodule.orchardbench_datamodule import OrchardBenchDataModule
import torch
traj=json.loads((bootstrap.ORCHARD/'data/orchard_v1_2650/filtered/json/val/episode_000010.json').read_text())
images=[Image.fromarray(VideoReader(traj['observations'][k][0]['path'],num_threads=1)[0].asnumpy()) for k in ('ego','wrist_left')]
processed=[prepare_rgb(im) for im in images]
p=traj['proprios']
sample=dict(messages=policy_messages(processed),state=torch.from_numpy(state_vector(p['ee_pos'][0],p['ee_rotm'][0],p['gripper_pos'][0],p['arm_joint'][0])),action=torch.zeros(30,32),action_mask=torch.from_numpy(action_mask()))
dm=OrchardBenchDataModule(dict(processor_path=str(bootstrap.XR0.parent/'checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D')))
batch=dm.collate_fn([sample])
result=dict(native_image_sizes=[list(im.size) for im in images],prepared_image_sizes=[list(im.size) for im in processed],image_grid_thw=batch['image_grid_thw'].tolist(),pixel_values_shape=list(batch['pixel_values'].shape),merge_size=dm.collate_fn.processor.image_processor.merge_size,patch_size=dm.collate_fn.processor.image_processor.patch_size,do_resize_in_collate=False,cuda_initialized=torch.cuda.is_initialized())
result['merged_image_tokens_per_view']=[int(np.prod(g)//(result['merge_size']**2)) for g in result['image_grid_thw']]
(root/'reach_plan/input_resolution_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
