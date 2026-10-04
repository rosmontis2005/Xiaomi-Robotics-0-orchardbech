#!/usr/bin/env python3
"""Measure existing deployment-vs-training generation differences, without dtype changes.

The evaluator calls run_probe on its already-loaded production OrchardPolicy.
The direct --validate entry reads CPU cache only and never constructs a model.
Reference and deployment receive exactly the same cached observation batch,
zero actions, no prefix and seed42; this excludes video/reset image differences.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import bootstrap

ROOT = bootstrap.ROOT


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_reset_samples():
    import torch
    selection_path = ROOT/'selection.json'
    selection = json.loads(selection_path.read_text())
    payload = torch.load(ROOT/'input_cache.pt', map_location='cpu', weights_only=True, mmap=True)
    if payload['schema'] != 'orchard_m0_cpu_input_cache_v1':
        raise ValueError('Unexpected CPU input-cache schema')
    if payload['provenance']['selection_sha256'] != sha256(selection_path):
        raise ValueError('CPU input cache belongs to another selection revision')
    samples = []
    for scene in selection['closed_loop_scenes']:
        candidates = [sample for sample in payload['samples']
                      if sample['meta']['episode_id'] == scene['episode_id']
                      and sample['meta']['split'] == scene['split']
                      and int(sample['meta']['frame']) == 0]
        if len(candidates) != 1:
            raise ValueError('Expected one cached reset window per selected scene')
        sample = candidates[0]
        if int(sample['meta']['seed']) != int(scene['seed']):
            raise ValueError('Cached reset seed mismatch')
        if sample['meta']['annotation_sha256'] != scene['annotation_sha256']:
            raise ValueError('Cached reset annotation hash mismatch')
        if not all(value.device.type == 'cpu' for value in sample['batch'].values()):
            raise ValueError('Input cache must contain only CPU tensors')
        if tuple(sample['batch']['action'].shape) != (1,30,32):
            raise ValueError('Unexpected cached action shape')
        if 'prefix_length' in sample['batch']:
            raise ValueError('Generation must not receive an action prefix')
        samples.append(sample)
    if len(samples) != 4:
        raise ValueError('Expected fixed 4-scene reset probe')
    return payload, samples


def validate_cpu():
    import torch
    before = torch.cuda.is_initialized()
    if before:
        raise RuntimeError('Direct CPU validation must run without initialized CUDA')
    payload, samples = load_reset_samples()
    zero_action_checks = []
    for sample in samples:
        batch = dict(sample['batch'])
        original = batch['action']
        batch['action'] = torch.zeros_like(original)
        assert not torch.count_nonzero(batch['action']).item()
        assert sample['batch']['action'] is original
        zero_action_checks.append(dict(window_id=sample['meta']['window_id'],
            seed=sample['meta']['seed'], action_shape=list(batch['action'].shape),
            generated_input_action_is_zero=True, original_cache_tensor_untouched=True))
    assert not torch.cuda.is_initialized()
    return dict(status='PASS', cuda_initialized=False, closed_loop_reset_windows=len(samples),
        selection_sha256=payload['provenance']['selection_sha256'],
        stats_sha256=payload['provenance']['stats_sha256'],
        source_cache=str(ROOT/'input_cache.pt'), checks=zero_action_checks,
        note='CPU validation did not load a model or generate predictions.')


def parameter_dtypes(model):
    groups = {}
    for label, vlm in (('non_vlm', False), ('vlm', True)):
        tensor_counts, element_counts = Counter(), Counter()
        for name, parameter in model.named_parameters():
            if name.startswith('vlm.') != vlm:
                continue
            dtype = str(parameter.dtype)
            tensor_counts[dtype] += 1
            element_counts[dtype] += parameter.numel()
        groups[label] = dict(tensors_by_dtype=dict(tensor_counts), elements_by_dtype=dict(element_counts))
    groups['action_output_layer_final_weight_dtype'] = str(model.action_output_layer.layers[2].weight.dtype)
    bias = model.action_output_layer.layers[2].bias
    groups['action_output_layer_final_bias_dtype'] = None if bias is None else str(bias.dtype)
    return groups


def array_difference(actual, reference):
    import numpy as np
    delta = np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
    return dict(mean_abs=float(delta.mean()), max_abs=float(delta.max()),
                per_dimension_mean_abs=delta.mean(axis=0).tolist(),
                per_dimension_max_abs=delta.max(axis=0).tolist(),
                exact_equal=bool(np.array_equal(actual, reference)))


def run_probe(policy, *, provider, reference_step, checkpoint_path, load_report):
    """Use the loaded production model unchanged; preserve torch RNG state."""
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from treesim.orchard_action import EPS
    if provider not in ('original_10k', 'best'):
        raise ValueError('Only model providers have a deployment precision probe')
    if policy.model.training:
        raise ValueError('Expected the unchanged production policy in eval mode')
    payload, samples = load_reset_samples()
    mean, std = payload['mean'].numpy(), payload['std'].numpy()
    if not np.array_equal(np.asarray(policy.mean), mean) or not np.array_equal(np.asarray(policy.std), std):
        raise ValueError('Policy and CPU panel normalization differ')
    panel_directory = ROOT/'evaluations'/f'step_{int(reference_step):04d}'
    reference_paths = [panel_directory/f"{s['meta']['split']}_{s['meta']['episode_id']}_frame0000.npz" for s in samples]
    for path in reference_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = ROOT/'deployment_precision'/provider
    output.mkdir(parents=True, exist_ok=False)
    dtypes_before = parameter_dtypes(policy.model)
    device = torch.device(policy.device)
    device_index = device.index or 0
    rows = []
    start = time.monotonic()
    with torch.random.fork_rng(devices=[device_index]), torch.inference_mode():
        for sample, reference_path in zip(samples, reference_paths):
            meta = sample['meta']
            # Do not transfer GT actions to the generation device, even temporarily.
            batch = {key: (torch.zeros_like(value) if key == 'action' else value).to(device)
                     for key, value in sample['batch'].items()}
            if torch.count_nonzero(batch['action']).item():
                raise AssertionError('Nonzero generation action input')
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = policy.model.generate(batch)[0]
            normalized = prediction.float().cpu().numpy()
            normalized[:,7:] = 0
            if not np.isfinite(normalized).all():
                raise FloatingPointError('Nonfinite deployment prediction')
            physical = normalized * (std + EPS) + mean
            physical[:,7:] = 0
            with np.load(reference_path, allow_pickle=False) as reference:
                matching = np.flatnonzero(reference['prediction_seeds'] == 42)
                if len(matching) != 1:
                    raise ValueError('Training panel must contain exactly one seed42 prediction')
                at = int(matching[0])
                expected_z = reference['normalized_prediction'][at].copy()
                expected_a = reference['predicted_action_physical'][at].copy()
            if normalized.shape != expected_z.shape or physical.shape != expected_a.shape:
                raise ValueError('Training/deployment prediction shape mismatch')
            norm_diff = array_difference(normalized[:,:7], expected_z[:,:7])
            physical_diff = array_difference(physical[:,:7], expected_a[:,:7])
            position_delta = np.linalg.norm(physical[:,:3] - expected_a[:,:3], axis=-1)
            rotation_delta = (Rotation.from_rotvec(expected_a[:,3:6]).inv()
                              * Rotation.from_rotvec(physical[:,3:6])).magnitude()
            width_delta = np.abs(physical[:,6] - expected_a[:,6])
            filename = f"{meta['split']}_{meta['episode_id']}_frame0000.npz"
            with (output/filename).open('xb') as stream:
                np.savez_compressed(stream, deployment_normalized=normalized, panel_normalized=expected_z,
                    deployment_physical=physical, panel_physical=expected_a,
                    position_difference_m=position_delta, rotation_difference_rad=rotation_delta,
                    width_difference_m=width_delta, prediction_seed=np.array(42))
            rows.append(dict(window_id=meta['window_id'], seed=meta['seed'], split=meta['split'],
                episode_id=meta['episode_id'], frame=0, panel_path=str(reference_path),
                panel_sha256=sha256(reference_path), result_npz=str(output/filename),
                normalized_active7_difference=norm_diff,
                physical_active7_component_difference=physical_diff,
                physical_position_difference_m=dict(mean=float(position_delta.mean()), max=float(position_delta.max())),
                physical_rotation_difference_rad=dict(mean=float(rotation_delta.mean()), max=float(rotation_delta.max())),
                physical_width_difference_m=dict(mean=float(width_delta.mean()), max=float(width_delta.max()))))
            del prediction, batch
    dtypes_after = parameter_dtypes(policy.model)
    if dtypes_before != dtypes_after:
        raise AssertionError('Probe changed parameter dtypes')
    config = json.loads((ROOT/'config.json').read_text())
    report = dict(provider=provider, checkpoint_path=str(checkpoint_path), reference_training_step=int(reference_step),
        load_report=load_report, generation_seed=42, generated_from_zero_action=True, prefix_supplied=False,
        model_parameter_dtypes=dtypes_before, parameter_dtypes_unchanged=True,
        training_config_precision=config.get('precision'), autocast_dtype='torch.bfloat16',
        observation_source='The same pre-collated CPU input_cache.pt batch as the training generation panel',
        actual_live_reset_input_used=False, selection_sha256=payload['provenance']['selection_sha256'],
        stats_sha256=payload['provenance']['stats_sha256'], model_num_steps=policy.model.num_steps,
        normalized_mean_abs_across_windows=float(np.mean([r['normalized_active7_difference']['mean_abs'] for r in rows])),
        normalized_max_abs_across_windows=max(r['normalized_active7_difference']['max_abs'] for r in rows),
        physical_component_mean_abs_across_windows=float(np.mean([r['physical_active7_component_difference']['mean_abs'] for r in rows])),
        physical_component_max_abs_across_windows=max(r['physical_active7_component_difference']['max_abs'] for r in rows),
        physical_component_aggregation_warning='Active7 mixes metres and radians; use separated physical position/rotation/width metrics for interpretation.',
        interpretation='Measures output differences between existing training-panel and production construction/precision; it does not change dtype or prove one execution is more accurate.',
        windows=rows, elapsed_seconds=time.monotonic()-start)
    with (output/'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return dict(path=str(output/'report.json'), provider=provider, reference_training_step=int(reference_step),
        windows=len(rows), model_parameter_dtypes=dtypes_before,
        normalized_mean_abs=report['normalized_mean_abs_across_windows'],
        normalized_max_abs=report['normalized_max_abs_across_windows'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true', required=True)
    parser.parse_args()
    result = validate_cpu()
    (ROOT/'deployment_precision_cpu_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
