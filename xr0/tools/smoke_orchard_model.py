"""Tiny fixed-batch/fixed-flow-noise diagnostic; never a model-quality evaluation."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from lightning import Trainer, Callback, seed_everything
from mibot.data.datamodule.orchardbench_datamodule import OrchardBenchDataModule
from mibot.data.datasets.orchardbench_dataset import OrchardBenchDataset
from mibot.models.runner.orchard_runner import OrchardRunner


class FixedNoise(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Hold flow t/noise fixed to make loss decrease interpretable on one sample.
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)


def main(args):
    seed_everything(42, workers=True)
    data = dict(processor_path=args.pretrained, num_workers=0,
                train_datasets=dict(root=args.root, stats_path=args.stats, batch_size=1))
    dm = OrchardBenchDataModule(data)
    ds = OrchardBenchDataset(data)
    loader = DataLoader(Subset(ds, [min(40,len(ds)-1)]), batch_size=1, collate_fn=dm.collate_fn)
    params = dict(pretrained=args.pretrained, freeze_vlm=True, diagnostic_path=args.output,
                  model=dict(type='XR0', vlm_config_path=str(Path(args.pretrained)/'config.json'),
                             training_repeat=1, enable_freq=False, async_train=False),
                  optimizer=dict(type='torch.optim.AdamW', params=dict(lr=1e-5, betas=(.9,.95),
                                 weight_decay=.1, eps=1e-8, foreach=False)),
                  scheduler=dict(type='torch.optim.lr_scheduler.ConstantLR',params=dict(factor=1.,total_iters=1)))
    runner = OrchardRunner(params)
    runner.configure_model()
    initial = runner.model.action_output_layer.layers[2].weight.detach().clone()
    trainer = Trainer(accelerator='gpu', devices=1, precision='bf16-mixed', max_steps=args.steps,
                      max_epochs=-1, logger=False, enable_checkpointing=False, callbacks=[FixedNoise()],
                      num_sanity_val_steps=0, limit_val_batches=0, gradient_clip_val=1.)
    trainer.fit(runner, train_dataloaders=loader)
    change = float((runner.model.action_output_layer.layers[2].weight.detach().cpu()-initial).abs().max())
    report = json.loads(Path(args.output).read_text())
    report.update(diagnostic='one fixed real window; identical flow timestep/noise every step',
                  action_output_weight_max_change=change,
                  loss_decreased=report['steps'][-1]['loss']<report['steps'][0]['loss'])
    assert change > 0 and report['loss_decreased']
    # Exact inactive-loss invariance and inactive output-gradient test.
    pred=torch.randn(1,30,32,device='cuda',requires_grad=True)
    target=torch.randn_like(pred);mask=torch.zeros_like(pred);mask[:,:,:7]=1
    loss=runner.model.compute_loss(pred,target,mask)['loss']
    changed=pred.detach().clone();changed[:,:,7:]+=1e4
    assert torch.equal(loss,runner.model.compute_loss(changed,target,mask)['loss'])
    loss.backward();assert torch.count_nonzero(pred.grad[:,:,7:])==0
    report['inactive_loss_and_gradient_test']='PASS'
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='steps'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--stats',required=True)
    p.add_argument('--pretrained',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=20)
    args=p.parse_args()
    if not 2 <= args.steps <= 50:p.error('Diagnostic is limited to 2-50 steps')
    main(args)
