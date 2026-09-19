"""Strict Orchard initialization and optional smoke evidence on the official runner."""
import json
from pathlib import Path
import torch
from mibot.models import MIMODEL
from mibot.models.runner.base_runner import BaseRunner
from mibot.utils.orchard_checkpoint import load_weights


@MIMODEL.register_module()
class OrchardRunner(BaseRunner):
    def __init__(self, params):
        super().__init__(params)
        self.freeze_vlm = params.get('freeze_vlm', True)
        self.diagnostic_path = params.get('diagnostic_path')
        self.evidence = {'steps': []}

    def configure_model(self):
        if hasattr(self, 'model'):
            return
        self.model = MIMODEL.build(self._model)
        if not self._pretrained:
            raise ValueError('Orchard baseline requires explicit pretrained weights')
        self.evidence['checkpoint'] = load_weights(self.model, self._pretrained)
        if self.freeze_vlm:
            self.model.vlm.requires_grad_(False)
            self.model.vlm.eval()
        # Keep master trainable weights/Adam moments FP32 at low LR; bf16-mixed
        # computes activations in bf16 without rounding away 1e-5 updates.
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        self.evidence['trainable_parameters'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('ORCHARD INIT', json.dumps(self.evidence), flush=True)

    def on_fit_start(self):
        if self.global_step != 0 or self.trainer.ckpt_path:
            raise ValueError('Use weights-only initialization, not a trainer resume')
        self.evidence['initial_global_step'] = self.global_step
        self.evidence['initial_optimizer_state_entries'] = len(self.trainer.optimizers[0].state)
        if self.diagnostic_path:
            self._initial_output_weight = self.model.action_output_layer.layers[2].weight.detach().cpu().clone()
            self.evidence['initial_scheduler_epoch'] = self.lr_schedulers().last_epoch

    def training_step(self, batch, batch_idx):
        if self.freeze_vlm:
            self.model.vlm.eval()
        result = super().training_step(batch, batch_idx)
        if not torch.isfinite(result['loss']):
            raise FloatingPointError('Nonfinite Orchard loss')
        self.current_loss = float(result['loss'].detach())
        return result

    def on_before_optimizer_step(self, optimizer):
        if not self.diagnostic_path:
            return
        gradients = [p.grad for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError('Missing/nonfinite gradients')
        self.evidence['steps'].append(dict(step=int(self.global_step), loss=self.current_loss,
            gradient_tensors=len(gradients), gradients_finite=True,
            gradient_max=max(float(g.abs().max()) for g in gradients), lr=optimizer.param_groups[0]['lr']))

    def on_train_end(self):
        if self.diagnostic_path:
            self.evidence['final_global_step'] = int(self.global_step)
            self.evidence['final_optimizer_state_entries'] = len(self.trainer.optimizers[0].state)
            self.evidence['final_scheduler_epoch'] = self.lr_schedulers().last_epoch
            self.evidence['output_weight_max_update'] = float((
                self.model.action_output_layer.layers[2].weight.detach().cpu() - self._initial_output_weight).abs().max())
            self.evidence['updated_parameters_finite'] = all(
                bool(torch.isfinite(p).all()) for p in self.model.parameters() if p.requires_grad)
            if not self.evidence['updated_parameters_finite']:
                raise FloatingPointError('Nonfinite parameter after optimizer step')
            self.evidence['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
            path = Path(self.diagnostic_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.evidence, indent=2, allow_nan=False) + '\n')
