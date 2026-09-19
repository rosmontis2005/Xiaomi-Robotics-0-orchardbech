"""Strict weights-only initialization. No optimizer/scheduler/global-step restore."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file


def load_weights(model, path):
    path = Path(path)
    if path.is_dir():
        index = json.loads((path / 'model.safetensors.index.json').read_text())['weight_map']
        state = {}
        for shard in sorted(set(index.values())):
            state.update(load_file(str(path / shard)))
        if set(state) != set(index):
            raise ValueError('Safetensors index and shard keys disagree')
        source_format = 'HF sharded safetensors'
    else:
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        source = payload.get('module', payload.get('state_dict', payload))
        state = {k.removeprefix('model.'): v for k, v in source.items()}
        source_format = 'PyTorch weights'
    # HF omits this tied tensor from safetensors; verify tying in the destination.
    embedding = 'vlm.model.language_model.embed_tokens.weight'
    head = 'vlm.lm_head.weight'
    if head not in state and embedding in state:
        if model.vlm.lm_head.weight is not model.vlm.get_input_embeddings().weight:
            raise ValueError('Cannot restore omitted lm_head unless destination ties embeddings')
        state[head] = state[embedding]
    # strict=True rejects every unexpected/missing tensor and shape mismatch.
    model.load_state_dict(state, strict=True)
    return dict(path=str(path.resolve()), format=source_format, tensors=len(state),
                strict=True, optimizer_restored=False, scheduler_restored=False)
