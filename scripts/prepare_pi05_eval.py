#!/usr/bin/env python3
"""Prepare plain PI0.5 offline-evaluation samples (soma-train TODO T4/T5, plan A).

Runs in the soma-fluxvla environment and reads episodes exactly the way training
does (the run's ParquetDataset class, its action window and video decoding), but
keeps the raw observation: absolute state[14], raw RGB images per camera, task
text, and the absolute ground-truth window [H, 14] with its mask. Inference and
scoring then run through piper-vla-deploy's real backend
(``piper-vla-deploy/scripts/evaluate_release.py``), so the reported MSE is what
deployment would produce.

Frames are sampled deterministically: every ``--stride``-th frame of every
episode of the chosen split, capped at ``--max-samples``.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np


def _hwc(frame):
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] == 3 and frame.shape[-1] != 3:
        frame = frame.transpose(1, 2, 0)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def select_rows(episode_index, frame_index, episodes, stride, horizon):
    """Dataset rows to evaluate: every stride-th frame, never an episode's last row."""
    episode_index = np.asarray(episode_index)
    frame_index = np.asarray(frame_index)
    wanted = set(int(e) for e in episodes)
    rows = []
    for ep in sorted(wanted):
        idx = np.flatnonzero(episode_index == ep)
        if len(idx) < 2:
            continue
        idx = idx[np.argsort(frame_index[idx])][:-1]  # the last row has no future action
        rows.extend(int(i) for i in idx[::stride])
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True, help='training run config.json')
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--split-manifest', type=Path, required=True)
    p.add_argument('--split-name', default='val', choices=('val', 'test', 'train'))
    p.add_argument('--stride', type=int, default=30)
    p.add_argument('--max-samples', type=int, default=200)
    p.add_argument('--output', type=Path, required=True, help='samples .npz to write')
    a = p.parse_args()

    from fluxvla import register_training_components
    register_training_components()
    from fluxvla.engines import TRANSFORMS, build_dataset_from_cfg

    training = json.loads(a.config.read_text())
    split = json.loads(a.split_manifest.read_text())
    if split.get('schema_version') != 'soma.train.split_manifest.v1':
        raise ValueError('split manifest must be soma.train.split_manifest.v1')
    episodes = split[a.split_name]
    if not episodes:
        raise ValueError(f'split {a.split_name!r} is empty')
    ds_cfg = training['train_dataloader']['dataset']
    leaf = copy.deepcopy(ds_cfg['datasets'][0])
    leaf['data_root_path'] = [str(a.dataset)]
    leaf['episode_indices'] = sorted(episodes)
    parse = [t for t in leaf['transforms'] if t['type'] == 'ProcessParquetInputs']
    if len(parse) != 1:
        raise ValueError('training config needs one ProcessParquetInputs')
    parse = TRANSFORMS.build(dict(parse[0], video_cache=None))
    leaf['transforms'] = []
    dataset = build_dataset_from_cfg(leaf)
    horizon = int(leaf['action_window_size'])
    columns = dataset.dataset.select_columns(['episode_index', 'frame_index'])
    rows = select_rows(columns['episode_index'], columns['frame_index'], episodes, a.stride, horizon)
    rows = rows[:a.max_samples]
    if not rows:
        raise ValueError('no frames selected')

    cams = [k.removeprefix('observation.images.') for k in parse.video_keys]
    out = {k: [] for k in ('state', 'gt_actions', 'gt_mask', 'episode_index', 'frame_index')}
    images = {c: [] for c in cams}
    tasks = []
    stats = ds_cfg['dataset_statistics']
    # __getitem__ maps through sample_indices; ask for the exact rows instead.
    position = {int(r): i for i, r in enumerate(dataset.sample_indices)}
    for row in rows:
        raw = dataset.__getitem__(position[row], stats)
        if int(raw['frame_index']) != int(columns['frame_index'][row]):
            raise RuntimeError(f'dataset resampled row {row}; refuse a silently different sample')
        sample = parse(copy.deepcopy(raw))
        out['state'].append(np.asarray(sample['states'], np.float32)[:14])
        out['gt_actions'].append(np.asarray(sample['actions'], np.float32)[:, :14])
        out['gt_mask'].append(np.asarray(raw['action_masks'], bool))
        out['episode_index'].append(int(raw['episode_index']))
        out['frame_index'].append(int(raw['frame_index']))
        for cam, frame in zip(cams, sample['images']):
            images[cam].append(_hwc(frame))
        tasks.append(str(raw.get('task_description') or raw.get('overall_task') or ''))
    meta = dict(schema_version='soma.pi05.eval_samples.v1', split_name=a.split_name,
                split_digest=split['split_digest'], split_policy=split['policy'], horizon=horizon,
                stride=a.stride, cameras=cams, tasks=tasks, config=str(a.config), dataset=str(a.dataset))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **{k: np.stack(v) if k in ('state', 'gt_actions', 'gt_mask') else np.asarray(v)
                          for k, v in out.items()},
             **{f'image_{c}': np.stack(v) for c, v in images.items()},
             meta=np.array(json.dumps(meta)))
    print(json.dumps(dict(samples=len(rows), episodes=sorted(set(out['episode_index'])), output=str(a.output))))


if __name__ == '__main__':
    main()
