"""A/B trainable-only checkpoints overlay an explicitly verified full 10k base."""
import bootstrap
import hashlib
import os
from pathlib import Path
import torch

FORMAT = 'orchard_m0_trainable_v1'
_HASH_CACHE = {}


def file_sha256(path):
    path = Path(path).resolve()
    st = path.stat(); key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _HASH_CACHE:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
        _HASH_CACHE[key] = h.hexdigest()
    return _HASH_CACHE[key]


def frozen_parameter_sha256(model):
    """Hash all VLM parameter bytes, names and dtypes; no buffers/KV caches."""
    h = hashlib.sha256()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not name.startswith('vlm.'): continue
            h.update(name.encode()); h.update(str(tuple(p.shape)).encode()); h.update(str(p.dtype).encode())
            flat = p.detach().reshape(-1)
            for chunk in flat.split(4 * 1024 * 1024):
                h.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def save_trainable_checkpoint(model, path, *, base_checkpoint, step, metadata):
    path = Path(path).resolve()
    if not path.is_relative_to(bootstrap.ROOT): raise ValueError('Checkpoint outside isolated run directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    active = {name: p.detach().cpu().float().clone() for name, p in model.named_parameters() if p.requires_grad}
    expected = {name for name, _ in model.named_parameters() if not name.startswith('vlm.')}
    if set(active) != expected: raise ValueError('Unexpected trainable key set; refuse partial VLM or missing head')
    payload = dict(format=FORMAT, base_checkpoint=str(Path(base_checkpoint).resolve()),
        base_sha256=file_sha256(base_checkpoint), step=int(step), trainable_keys_sorted=sorted(active),
        trainable_state_dict=active, metadata=metadata)
    temporary = path.with_name(path.name + '.tmp')
    if temporary.exists(): raise FileExistsError(temporary)
    with temporary.open('xb') as f: torch.save(payload, f)
    os.replace(temporary, path)
    return dict(path=str(path), step=int(step), format=FORMAT, base_checkpoint=payload['base_checkpoint'],
                base_sha256=payload['base_sha256'], trainable_tensors=len(active),
                trainable_parameters=sum(x.numel() for x in active.values()), bytes=path.stat().st_size,
                checkpoint_sha256=file_sha256(path))


def load_trainable_overlay(model, checkpoint_path, *, base_checkpoint):
    """Call after loading full base weights; independent of requires_grad flags.

    Copying this FP32 overlay preserves the destination model parameter dtypes.
    The loader reports actual destination dtype counts rather than assuming a
    particular training or inference construction path.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
    if payload.get('format') != FORMAT: raise ValueError('Unexpected trainable checkpoint format')
    actual_base = Path(base_checkpoint).resolve()
    if actual_base != Path(payload['base_checkpoint']).resolve(): raise ValueError('Wrong base checkpoint path')
    if file_sha256(actual_base) != payload['base_sha256']: raise ValueError('Base checkpoint checksum mismatch')
    params = dict(model.named_parameters())
    expected = {k for k in params if not k.startswith('vlm.')}
    state = payload['trainable_state_dict']
    if set(state) != expected or payload['trainable_keys_sorted'] != sorted(expected):
        raise ValueError('Overlay keys must exactly equal all non-VLM model parameters')
    for name, value in state.items():
        if value.shape != params[name].shape or value.dtype != torch.float32:
            raise ValueError(f'Invalid overlay shape/dtype: {name}')
        if not torch.isfinite(value).all(): raise FloatingPointError(f'Nonfinite overlay: {name}')
    with torch.no_grad():
        for name, value in state.items(): params[name].copy_(value)
    return dict(path=str(checkpoint_path), format=FORMAT, step=payload['step'],
                base_checkpoint=str(actual_base), base_sha256=payload['base_sha256'],
                checkpoint_sha256=file_sha256(checkpoint_path), tensors=len(state),
                parameters=sum(t.numel() for t in state.values()), exact_keys=True,
                destination_dtype_counts={str(dtype): sum(params[n].dtype == dtype for n in state)
                                          for dtype in {params[n].dtype for n in state}},
                metadata=payload.get('metadata', {}))
