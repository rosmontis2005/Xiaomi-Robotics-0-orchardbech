"""Orchard loader through the official Lightning/CustomCollate pipeline."""
from lightning import LightningDataModule
from mmengine import DATASETS
from torch.utils.data import DataLoader
from transformers import Qwen3VLProcessor
from mibot.data.collate.custom_collate import CustomCollate


@DATASETS.register_module()
class OrchardBenchDataModule(LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        # Explicit class: the HF CALVIN AutoProcessor includes CALVIN action stats.
        self.collate_fn = CustomCollate.__new__(CustomCollate)
        self.collate_fn.processor = Qwen3VLProcessor.from_pretrained(params['processor_path'], local_files_only=True)
        self.collate_fn.processor.tokenizer.padding_side = 'right'

    def train_dataloader(self):
        from mibot.data.datasets.orchardbench_dataset import OrchardBenchDataset
        workers = int(self.params.get('num_workers', 2))
        return DataLoader(OrchardBenchDataset(self.params),
                          batch_size=self.params['train_datasets']['batch_size'], shuffle=True,
                          num_workers=workers, persistent_workers=workers > 0,
                          pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return []
