"""Deterministic population statistics over every accepted TRAIN full window."""
import argparse
import json
from pathlib import Path
import numpy as np
from treesim.orchard_action import CONTRACT, HORIZON, EPS
from mibot.data.datasets.orchardbench_dataset import OrchardBenchDataset, fingerprint


def compute(root, output, episode_ids=None):
    dataset = OrchardBenchDataset({'train_datasets': dict(root=root, split='train', episode_ids=episode_ids)}, normalize=False)
    mean = np.zeros((HORIZON, 32), dtype=np.float64)
    m2 = np.zeros_like(mean)
    # Welford avoids cancellation in small near-constant physical dimensions.
    for count in range(1, len(dataset) + 1):
        x = dataset.raw_action(count - 1).astype(np.float64)
        delta = x - mean
        mean += delta / count
        m2 += delta * (x - mean)
    std = np.sqrt(m2 / count)
    std[:, :7] = np.maximum(std[:, :7], 1e-4)
    mean[:, 7:] = 0
    std[:, 7:] = 1
    result = dict(contract=CONTRACT, source_split='train', source_sha256=fingerprint(dataset.files),
                  episodes=[Path(p).stem for p in dataset.files], full_window_samples=len(dataset),
                  action_length=HORIZON, normalization_epsilon=EPS, active_std_floor=1e-4,
                  mean=mean.astype(np.float32).tolist(), std=std.astype(np.float32).tolist())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('mean', 'std', 'episodes')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--episode-ids', help='Optional JSON list of accepted train IDs for ablations')
    args = parser.parse_args()
    compute(args.root, args.output, args.episode_ids)
