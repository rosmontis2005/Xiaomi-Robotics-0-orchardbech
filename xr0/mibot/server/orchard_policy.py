"""Local Orchard inference bridge. Does not use CALVIN processor action statistics.

On each replan: policy.predict(obs); then env.step(**policy.command(k, obs)).
The chunk anchor is captured once per prediction, current pose once per command.
"""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from treesim.orchard_action import OrchardActionAdapter, action_mask, prepare_rgb
from mibot.data.datasets.orchardbench_dataset import load_stats, policy_messages
from mibot.data.datamodule.orchardbench_datamodule import OrchardBenchDataModule
from mibot.models import MIMODEL
from mibot.utils.orchard_checkpoint import load_weights


class OrchardPolicy:
    def __init__(self, weights, processor_path, stats_path, device='cuda'):
        self.device = device
        self.stats, self.mean, self.std = load_stats(stats_path)
        self.model = MIMODEL.build(dict(type='XR0', vlm_config_path=str(Path(processor_path)/'config.json'),
                                       async_train=False, enable_freq=False, training_repeat=1))
        self.load_report = load_weights(self.model, weights)
        self.model.eval().to(device)
        self.collate = OrchardBenchDataModule(dict(processor_path=processor_path)).collate_fn
        self.adapter = OrchardActionAdapter()

    @torch.inference_mode()
    def predict(self, obs, seed=42):
        images = [prepare_rgb(Image.fromarray(obs[k])) for k in ('rgb_static','rgb_wrist')]
        sample = dict(messages=policy_messages(images), state=torch.from_numpy(self.adapter.state(obs)),
                      action=torch.zeros(30,32), action_mask=torch.from_numpy(action_mask()))
        batch = {k:v.to(self.device) for k,v in self.collate([sample]).items()}
        with torch.random.fork_rng(devices=[torch.device(self.device).index or 0]):
            torch.manual_seed(seed)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                normalized = self.model.generate(batch)[0].float().cpu().numpy()
        normalized[:,7:] = 0
        if not np.isfinite(normalized).all():
            raise FloatingPointError('Nonfinite Orchard prediction')
        self.adapter.set_chunk(normalized,self.mean,self.std,obs)
        return normalized

    def command(self, index, obs):
        return self.adapter.to_native(index,obs)
