"""Build our FAST/subtask fixture from a LeRobot v2.1 export (fabric sort dataset).

The export carries per-frame `subtask`, `overall_task` and `subtask_valid` columns, so no
external annotation is needed. Produces `samples.npz` (uncompressed, so training can
memory-map images) and `samples.json`, in the layout `run_fast_stage1.py` expects.

    python scripts/architecture/prepare_lerobot_fixture.py \
        --dataset <lerobot v2.1 dir> --output <fixture dir> \
        --val-episodes 2 --anchors-every 15 --horizon 50

Conventions (match the shoes pipeline so results stay comparable):
  * state/action are quantile-normalised per dimension using q01/q99 fitted on TRAIN
    episodes only, then zero-padded from 14 to 32 dimensions.
  * joint dimensions are stored as deltas to the state at the anchor frame; gripper
    dimensions stay absolute.
  * an anchor is kept only when the whole 50-frame action window lies inside one
    `subtask_valid` run, so no window crosses a subtask boundary.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

IMAGE_KEYS = ('images.cam_front', 'images.cam_left_wrist', 'images.cam_right_wrist')
JOINT_DIMS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)   # 6 joints per arm; 6/13 = grippers
MAX_ACTION_DIM = 32
IMAGE_SIZE = 224


def decode_image(cell):
    if isinstance(cell, dict):
        payload = cell.get('bytes')
        if payload is None:
            raise ValueError('image cell has no bytes payload')
    else:
        payload = cell
    return np.asarray(Image.open(io.BytesIO(payload)).convert('RGB'))


def resize(frame):
    """Return CHW uint8 so the three cameras stack into 9 channels (N,9,H,W)."""
    out = Image.fromarray(frame).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(out).transpose(2, 0, 1)


def runs_of(subtask, valid):
    """Yield (start, end, label) for maximal runs of one valid subtask label."""
    start = None
    label = None
    for i, (text, ok) in enumerate(zip(subtask, valid)):
        if ok:
            if start is None:
                start, label = i, text
            elif text != label:
                yield start, i, label
                start, label = i, text
        else:
            if start is not None:
                yield start, i, label
                start = None
    if start is not None:
        yield start, len(subtask), label


def quantile_stats(values):
    low = np.quantile(values, 0.01, axis=0)
    high = np.quantile(values, 0.99, axis=0)
    return dict(q01=low.tolist(), q99=high.tolist())


def normalise(values, stats):
    low = np.asarray(stats['q01'], dtype=np.float32)
    high = np.asarray(stats['q99'], dtype=np.float32)
    span = np.where(high - low < 1e-6, 1.0, high - low)
    return (2 * (values - low) / span - 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--val-episodes', default='', help='comma separated episode indices')
    ap.add_argument('--anchors-every', type=int, default=15)
    ap.add_argument('--horizon', type=int, default=50)
    ap.add_argument('--limit-anchors', type=int, default=0, help='debug: cap anchors per episode')
    ap.add_argument('--image-source', choices=('auto', 'parquet', 'video'), default='auto',
                    help='v2.1 exports store frames in videos/; older ones inline in parquet')
    args = ap.parse_args()

    root = Path(args.dataset)
    info = json.loads((root / 'meta/info.json').read_text())
    episodes = [json.loads(line) for line in (root / 'meta/episodes.jsonl').read_text().splitlines() if line]
    val = {int(x) for x in args.val_episodes.split(',') if x.strip() != ''}
    horizon = args.horizon

    # Pass 1: metadata columns only (small). Images are decoded per row group in pass 2 so
    # peak memory stays bounded on multi-GB exports.
    features = info['features']
    has_inline = any(k in features and features[k].get('dtype') == 'image' for k in IMAGE_KEYS)
    video_dir = root / 'videos'
    image_source = args.image_source
    if image_source == 'auto':
        image_source = 'parquet' if has_inline else ('video' if video_dir.is_dir() else 'parquet')
    if image_source == 'video' and not video_dir.is_dir():
        raise SystemExit(f'{root}: --image-source video but no videos/ directory')
    print(f'image source: {image_source}', flush=True)

    def video_path(episode, key):
        chunk = episode // info['chunks_size']
        return video_dir / f'chunk-{chunk:03d}' / key / f'episode_{episode:06d}.mp4'

    def decode_video_frames(episode, frames, expect_rows):
        """Decode the requested frames from every camera in one batched read."""
        out = []
        for key in IMAGE_KEYS:
            path = video_path(episode, key)
            if not path.exists():
                raise SystemExit(f'missing video {path}')
            import decord
            reader = decord.VideoReader(str(path), num_threads=2)
            if len(reader) != expect_rows:
                raise SystemExit(f'{path}: {len(reader)} frames but parquet has {expect_rows} rows; '
                                 'refusing to guess the frame mapping')
            batch = reader.get_batch(sorted(frames)).asnumpy()
            out.append({frame: batch[i] for i, frame in enumerate(sorted(frames))})
        return out

    per_episode, anchors = {}, {}
    for entry in episodes:
        idx = int(entry['episode_index'])
        path = root / info['data_path'].format(episode_chunk=idx // info['chunks_size'],
                                               episode_index=idx)
        table = pq.read_table(path, columns=['state', 'actions', 'subtask', 'overall_task',
                                             'subtask_valid', 'timestamp']).to_pydict()
        per_episode[idx] = dict(path=path, table=table)
        subtask = [str(x) for x in table['subtask']]
        valid = [int(x) != 0 for x in table['subtask_valid']]
        picked = []
        for start, end, label in runs_of(subtask, valid):
            anchor = start
            while anchor + horizon < end:
                if args.limit_anchors and len(picked) >= args.limit_anchors:
                    break
                picked.append((anchor, label))
                anchor += args.anchors_every
            if args.limit_anchors and len(picked) >= args.limit_anchors:
                break
        anchors[idx] = picked
        print(f'episode {idx}: frames={len(table["state"])} anchors={len(picked)}', flush=True)

    train_eps = [i for i in per_episode if i not in val]
    if not train_eps:
        raise SystemExit('no training episodes; check --val-episodes')

    # Train-only normalisation statistics.
    train_states = np.concatenate([np.asarray(per_episode[i]['table']['state'], dtype=np.float32)
                                   for i in train_eps])
    train_actions = np.concatenate([np.asarray(per_episode[i]['table']['actions'], dtype=np.float32)
                                    for i in train_eps])
    stats = dict(state=quantile_stats(train_states), action=quantile_stats(train_actions))

    images, states_out, actions_out, masks, records = [], [], [], [], []
    for idx, entry in per_episode.items():
        table = entry['table']
        state = np.asarray(table['state'], dtype=np.float32)
        actions = np.asarray(table['actions'], dtype=np.float32)
        overall = [str(x) for x in table['overall_task']]
        stamps = [float(x) for x in table['timestamp']]
        if len(set(overall)) != 1:
            raise SystemExit(f'episode {idx} has more than one overall task')
        split = 'validation' if idx in val else 'train'
        picked = dict(anchors[idx])
        # Pass 2: only decode the anchor frames.
        if image_source == 'video':
            decoded = decode_video_frames(idx, list(picked), len(state))
            for frame in sorted(picked):
                cams = [resize(decoded[cam][frame]) for cam in range(len(IMAGE_KEYS))]
                images.append(np.concatenate(cams, axis=0))
        else:
            parquet = pq.ParquetFile(entry['path'])
            offset = 0
            for rg in range(parquet.num_row_groups):
                rows = parquet.metadata.row_group(rg).num_rows
                inside = sorted(f for f in picked if offset <= f < offset + rows)
                if inside:
                    group = parquet.read_row_group(rg, columns=list(IMAGE_KEYS)).to_pydict()
                    for frame in inside:
                        cams = [resize(decode_image(group[key][frame - offset])) for key in IMAGE_KEYS]
                        images.append(np.concatenate(cams, axis=0))
                offset += rows
        for frame, label in sorted(picked.items()):
            window = actions[frame + 1:frame + 1 + horizon].copy()
            window[:, JOINT_DIMS] -= state[frame, JOINT_DIMS]
            padded = np.zeros((horizon, MAX_ACTION_DIM), dtype=np.float32)
            padded[:, :window.shape[1]] = normalise(window, stats['action'])
            mask = np.zeros((horizon, MAX_ACTION_DIM), dtype=bool)
            mask[:, :window.shape[1]] = True
            actions_out.append(padded)
            masks.append(mask)
            states_out.append(normalise(state[frame][None], stats['state'])[0])
            parts = [part.strip() for part in label.split('||') if part.strip()]
            records.append(dict(sample_id=f'ep{idx:03d}_f{frame:05d}', episode=idx,
                                frame=frame, timestamp=stamps[frame], split=split,
                                stage=label, subtask=label, overall_task=overall[frame],
                                is_parallel=len(parts) > 1, parallel_parts=parts,
                                annotation_source='dataset_subtask_column'))
        print(f'episode {idx}: kept={len(picked)} split={split}', flush=True)

    if not images:
        raise SystemExit('no anchors built')

    payload = dict(images=np.stack(images), states=np.stack(states_out),
                   actions=np.stack(actions_out), action_masks=np.stack(masks),
                   img_masks=np.ones((len(images), len(IMAGE_KEYS)), dtype=bool))
    if not all(np.isfinite(payload[k]).all() for k in ('states', 'actions')):
        raise SystemExit('non-finite values in fixture')

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'samples.npz', **payload)          # uncompressed: enables mmap
    manifest = dict(
        records=records, normalization=stats,
        dataset=str(root.resolve()),
        dataset_info=dict(codebase_version=info['codebase_version'], fps=info['fps'],
                          total_episodes=info['total_episodes'], total_frames=info['total_frames']),
        val_episodes=sorted(val), anchors_every=args.anchors_every, horizon=horizon,
        arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in payload.items()},
        conventions=dict(delta_joint_dims=list(JOINT_DIMS), max_action_dim=MAX_ACTION_DIM,
                         image_size=IMAGE_SIZE, cameras=list(IMAGE_KEYS),
                         image_source=image_source,
                         normalization='q01/q99 from train episodes, 14 dims, zero-padded to 32'),
        samples=len(records),
        split_counts={s: sum(r['split'] == s for r in records) for s in ('train', 'validation')},
        distinct_subtasks=sorted({r['subtask'] for r in records}),
        parallel_samples=sum(1 for r in records if r.get('is_parallel')),
    )
    (out / 'samples.json').write_text(json.dumps(manifest, indent=2))
    digest = hashlib.sha256((out / 'samples.npz').read_bytes()).hexdigest()
    print(json.dumps(dict(samples=len(records), split_counts=manifest['split_counts'],
                          distinct_subtasks=len(manifest['distinct_subtasks']),
                          npz_sha256=digest, bytes=(out / 'samples.npz').stat().st_size), indent=2))


if __name__ == '__main__':
    main()
