#!/usr/bin/env python3
"""GPU-slot-only paired B8000 reference on the new auxiliary panel."""
import bootstrap
import fcntl
import json
import os
from pathlib import Path
import time
import torch
import numpy as np
import train_l as training
import ab_common as common
from checkpoint_io import file_sha256,load_trainable_overlay,frozen_parameter_sha256


def main():
    training.check_prepared();torch.set_num_threads(1)
    experiment_lock=(bootstrap.EXPERIMENT/'.gpu.lock').open('a+');fcntl.flock(experiment_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    historical_lock=(bootstrap.AB/'.gpu_training.lock').open('r');fcntl.flock(historical_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=(bootstrap.ROOT/'.gpu_training.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=bootstrap.ROOT/'reference_B8000_extra';out.mkdir(exist_ok=True)
    if (out/'completed.json').exists():raise FileExistsError('Reference already complete')
    started=time.monotonic();model,initial=training.construct_model()
    checkpoint=bootstrap.AB/'arms/B/checkpoints/step_8000_trainable.pt'
    report=load_trainable_overlay(model,checkpoint,base_checkpoint=common.BASE)
    historical=json.loads((bootstrap.AB/'arms/B/training_summary.json').read_text())
    assert historical['status']=='COMPLETE' and historical['completed_updates']==8000
    assert report['checkpoint_sha256']==historical['checkpoint_records']['8000']['checkpoint_sha256']
    assert report['step']==8000 and report['metadata']['provenance']['arm']=='B'
    assert report['metadata']['provenance']['selection_sha256']==file_sha256(common.SELECTION)
    assert all(p.dtype==torch.float32 for n,p in model.named_parameters() if not n.startswith('vlm.'))
    frozen=frozen_parameter_sha256(model);model.to('cuda:0');model.eval();model.vlm.eval()
    panel=torch.load(training.AUX_CACHE,map_location='cpu',weights_only=True,mmap=True)
    result=common.evaluate(model,panel['samples'],panel['mean'].numpy(),panel['std'].numpy(),8000,'cuda:0',out,apply_fit_gate=False)
    assert result['windows']==144 and result['fit_gate'] is None
    overlaps=0;arrays=0
    for current in (out/'evaluations/step_8000').glob('*.npz'):
        reference=bootstrap.AB/'arms/B/evaluations/step_8000'/current.name
        if not reference.exists():continue
        overlaps+=1
        with np.load(current) as a,np.load(reference) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:
                assert np.array_equal(a[key],b[key]),(current.name,key)
                arrays+=1
    assert overlaps==49
    common.write_json(out/'completed.json',dict(status='COMPLETE',pid=os.getpid(),windows=144,seeds=[42,43,44],
        overlap_old_panel_windows=overlaps,overlap_arrays_exact=arrays,elapsed_s=time.monotonic()-started,initial_load=initial,overlay=report,frozen_vlm_parameter_sha256=frozen,
        trainable_dtype='float32',prepared_checks_sha256=file_sha256(bootstrap.ROOT/'prepared_checks.json'),
        aux_cache_sha256=file_sha256(training.AUX_CACHE),source_sha256=training.source_hashes(),
        report_sha256=file_sha256(out/'evaluations/step_8000/report.json')))
    print(json.dumps(dict(event='reference_complete',directory=str(out))),flush=True)

if __name__=='__main__':main()
