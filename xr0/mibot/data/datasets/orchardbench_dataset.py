"""Accepted-manifest-only, single FR3 Cartesian dataset. No debug model inputs."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset

from treesim.orchard_action import CONTRACT, HORIZON, action_mask, encode_window, prepare_rgb, state_vector
from mibot.utils.io import normalize_action, validate_stats
from mibot.data.datasets.json_dataset import JsonDataset


@lru_cache(maxsize=32)
def read_episode(path):
    return json.loads(Path(path).read_text())


def episode_files(root, split='train', episode_ids=None):
    root = Path(root).resolve()
    rows = [json.loads(line) for line in (root / 'manifest.jsonl').read_text().splitlines()]
    accepted = [r for r in rows if r['accepted']]
    if len({r['episode_id'] for r in accepted}) != len(accepted):
        raise ValueError('Duplicate accepted episode IDs')
    chosen = [r for r in accepted if r['split'] == split]
    if episode_ids is not None:
        ids = set(json.loads(Path(episode_ids).read_text()))
        if not ids <= {r['episode_id'] for r in chosen}:
            raise ValueError('Subset must contain only accepted IDs from the requested split')
        chosen = [r for r in chosen if r['episode_id'] in ids]
    files = []
    for row in chosen:
        path = root / row['annotation']
        t = read_episode(str(path))
        if t['episode_id'] != row['episode_id'] or t['split'] != split:
            raise ValueError(f'Manifest/annotation mismatch: {path}')
        validate_episode(t)
        files.append(str(path))
    if not files:
        raise ValueError(f'No accepted {split} episodes in {root}')
    return files


def validate_episode(t):
    if t['schema'] != 'orchard_xr0_single_arm_v0' or t['trajectory_type'] != 'success' or t['record_fps'] != 30:
        raise ValueError('Expected accepted Orchard 30 Hz schema')
    n = t['num_frames']
    dims = dict(ee_pos=3, ee_rotm=9, arm_joint=7, arm_joint_vel=7, gripper_pos=1)
    for group in ('proprios', 'actions'):
        expected = set(dims) - ({'arm_joint_vel'} if group == 'actions' else set())
        if set(t[group]) != expected:
            raise ValueError(f'Unexpected fields in {group}')
        for key, value in t[group].items():
            a = np.asarray(value)
            if a.shape != (n, dims[key]) or not np.isfinite(a).all():
                raise ValueError(f'Invalid {group}.{key}')
    m = t['orchardbench']
    if (not all(m[k] for k in ('first_attempt_success', 'grasped', 'detached', 'placed'))
            or m['incidental_detach_count'] or m['detach_diagnostics']['premature_detach'] is not False
            or m['base_policy'] != 'privileged_reset_stance_fixed_base_expert_v1_world_weld'):
        raise ValueError('Not an accepted fixed-base V1 demonstration')
    if set(t['observations']) != {'ego', 'wrist_left'}:
        raise ValueError('Expected exactly static and wrist RGB views')


def fingerprint(files):
    h = hashlib.sha256()
    for path in files:
        h.update(Path(path).name.encode())
        h.update(Path(path).read_bytes())
    return h.hexdigest()


def load_stats(path):
    stats = json.loads(Path(path).read_text())
    if stats['contract'] != CONTRACT or stats['source_split'] != 'train':
        raise ValueError('Wrong normalization contract or source split')
    mean, std = validate_stats(stats['mean'], stats['std'], HORIZON)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Invalid normalization values')
    if np.any(mean[:, 7:] != 0) or np.any(std[:, 7:] != 1):
        raise ValueError('Inactive dimensions require mean=0, std=1')
    return stats, mean, std


class OrchardBenchDataset(Dataset):
    def __init__(self, params, *, normalize=True):
        data = params['train_datasets']
        if int(data.get('action_length', HORIZON)) != HORIZON:
            raise ValueError('Initial Orchard baseline requires action_length=30')
        self.root = Path(data['root']).resolve()
        self.split = data.get('split', 'train')
        self.files = episode_files(self.root, self.split, data.get('episode_ids'))
        self.samples = [(path, frame) for path in self.files
                        for frame in range(read_episode(path)['num_frames'] - HORIZON + 1)]
        if not self.samples:
            raise ValueError('No full windows')
        self.normalize = normalize
        if normalize:
            self.stats, self.mean, self.std = load_stats(data['stats_path'])
            # Validation consumes the training stats, never computes its own.
            train_files = self.files if self.split == 'train' else episode_files(self.root, 'train')
            if self.stats['source_sha256'] != fingerprint(train_files):
                raise ValueError('Statistics source differs from training selection; recompute stats')

    def __len__(self):
        return len(self.samples)

    def raw_action(self, index):
        path, frame = self.samples[index]
        return encode_window(read_episode(path), frame)

    def __getitem__(self, index):
        path, frame = self.samples[index]
        t = read_episode(path)
        p = t['proprios']
        images = []
        for key in ('ego', 'wrist_left'):
            info = t['observations'][key][0]
            video_path = Path(info['path'])
            if not video_path.is_absolute():
                video_path = self.root / video_path
            # Allows relocating a mounted canonical dataset without editing its JSON.
            if not video_path.exists():
                video_path = self.root / 'videos' / Path(info['path']).name
            video = VideoReader(str(video_path), num_threads=1)
            if len(video) != t['num_frames']:
                raise ValueError(f'Video/annotation frame mismatch: {video_path}')
            images.append(prepare_rgb(Image.fromarray(video[frame].asnumpy())))
        action = self.raw_action(index)
        if self.normalize:
            action = normalize_action(action, self.mean, self.std)
        return dict(messages=policy_messages(images),
                    action=torch.from_numpy(action), action_mask=torch.from_numpy(action_mask()),
                    state=torch.from_numpy(state_vector(p['ee_pos'][frame], p['ee_rotm'][frame],
                                                        p['gripper_pos'][frame], p['arm_joint'][frame])))


def policy_messages(images):
    # Whitelist the exact policy prompt, never concatenate metadata fields.
    prompt = ('The following observations are captured from multiple views.\n'
              '# Base View\n<image>\n# Left-Wrist View\n<image>\n'
              'Generate robot actions for the task:\nPick an apple and place it in the bucket. /no_cot')
    conversations = [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': '<cot></cot>'}]
    return JsonDataset._messages(conversations, images)
