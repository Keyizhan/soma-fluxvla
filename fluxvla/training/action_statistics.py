"""Fit PI0.5 action statistics in the actual post-delta training domain.

No torch/model imports: fitting is CPU-only and the contract loader is stdlib-only.
Padded targets are excluded. Every valid target of every training anchor is counted,
including the shorter valid windows at episode tails.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

JOINTS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
GRIPPERS = (6, 13)
SCHEMA = 'soma.pi05.delta_statistics.v1'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def contract_path(path):
    return Path(path).with_suffix('.contract.json')


def target_windows(states, actions, horizon=50, window_start=1, batch_size=512):
    """Yield the unpadded targets produced by ProcessLiberoActions.

    ParquetDatasetV3 skips the last episode row as an anchor, and starts the
    action window at anchor+1. Grippers stay absolute metres; all 12 joints
    (including wrist joints 6) are relative to the anchor's measured state.
    """
    import numpy as np

    if horizon < 1 or window_start != 1:
        raise ValueError('require positive horizon and window_start=1')
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if states.shape != actions.shape or states.ndim != 2 or states.shape[1] != 14:
        raise ValueError('require aligned [N,14] states/actions')
    offsets = np.arange(window_start, window_start + horizon)
    for start in range(0, len(states) - 1, batch_size):
        anchors = np.arange(start, min(start + batch_size, len(states) - 1))
        indices = anchors[:, None] + offsets[None, :]
        valid = indices < len(states)
        targets = actions[np.minimum(indices, len(states) - 1)].copy()
        targets[:, :, JOINTS] -= states[anchors, None, :][:, :, JOINTS]
        yield targets[valid]


def _statistics(values):
    import numpy as np

    result = {key: [] for key in ('min', 'max', 'mean', 'std', 'q01', 'q99')}
    # Copy/partition one dimension at a time rather than duplicating the full mmap.
    for dim in range(14):
        channel = np.array(values[:, dim], copy=True)
        for key, value in (
            ('min', channel.min()), ('max', channel.max()),
            ('mean', channel.mean(dtype=np.float64)),
            ('std', channel.std(dtype=np.float64)),
        ):
            result[key].append(float(value))
        q01, q99 = np.quantile(channel, [.01, .99], method='linear')
        result['q01'].append(float(q01))
        result['q99'].append(float(q99))
    return result


def fit(dataset, episodes, output, *, horizon=50, window_start=1, scratch=None):
    import numpy as np
    import pyarrow.parquet as pq

    root, output = Path(dataset).resolve(), Path(output)
    episodes = sorted(set(int(ep) for ep in episodes))
    if not episodes or horizon < 1 or window_start != 1:
        raise ValueError('require training episodes, positive horizon and window_start=1')
    if output.exists() or contract_path(output).exists():
        raise FileExistsError('statistics are immutable; choose a new output filename')
    info = json.loads((root / 'meta/info.json').read_text())
    paths, counts, sources = [], [], []
    for ep in episodes:
        path = root / info['data_path'].format(
            episode_index=ep, episode_chunk=ep // int(info.get('chunks_size', 1000)))
        n = pq.ParquetFile(path).metadata.num_rows
        if n < 2:
            raise ValueError(f'episode {ep}: at least two frames required')
        paths.append(path)
        counts.append(n)
        sources.append({'episode': ep, 'path': str(path.relative_to(root)),
                        'frames': n, 'sha256': digest(path)})
    state_rows = sum(counts)
    action_rows = sum(sum(max(0, n - offset) for offset in range(1, horizon + 1))
                      for n in counts)
    # The mmap scratch is bounded on disk; no video/model is read.
    with tempfile.TemporaryDirectory(prefix='pi05-delta-stats-', dir=scratch) as temp:
        states_map = np.lib.format.open_memmap(
            Path(temp) / 'states.npy', mode='w+', dtype=np.float32, shape=(state_rows, 14))
        actions_map = np.lib.format.open_memmap(
            Path(temp) / 'targets.npy', mode='w+', dtype=np.float32, shape=(action_rows, 14))
        state_at, action_at = 0, 0
        for ep, path, n in zip(episodes, paths, counts):
            names = pq.ParquetFile(path).schema_arrow.names
            required = ['observation.state', 'action', 'frame_index', 'episode_index', 'timestamp']
            columns = required + [key for key in ['overall_task'] if key in names]
            data = pq.read_table(path, columns=columns).to_pydict()
            states = np.asarray(data['observation.state'], dtype=np.float32)
            actions = np.asarray(data['action'], dtype=np.float32)
            if states.shape != (n, 14) or actions.shape != (n, 14):
                raise ValueError(f'episode {ep}: require 14D state/action')
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise ValueError(f'episode {ep}: nonfinite trajectories')
            if not np.array_equal(data['frame_index'], np.arange(n)):
                raise ValueError(f'episode {ep}: noncontiguous frame indices')
            if set(data['episode_index']) != {ep}:
                raise ValueError(f'episode {ep}: inconsistent episode indices')
            timestamps = np.asarray(data['timestamp'], dtype=np.float64)
            if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
                raise ValueError(f'episode {ep}: non-increasing timestamps')
            if any(str(task).strip() in ('', 'empty', 'static')
                   for task in data.get('overall_task', [])):
                raise ValueError('empty/static task filtering is not supported by this fitter')
            states_map[state_at:state_at + n] = states
            state_at += n
            for targets in target_windows(states, actions, horizon, window_start):
                actions_map[action_at:action_at + len(targets)] = targets
                action_at += len(targets)
            print(f'STATS_EPISODE {ep}: states={state_at}, valid_targets={action_at}', flush=True)
        if state_at != state_rows or action_at != action_rows:
            raise RuntimeError('target counting disagrees with window assembly')
        statistics = {'private': {'proprio': _statistics(states_map),
                                  'action': _statistics(actions_map)}}
        del states_map, actions_map
    content = (json.dumps(statistics, indent=2) + '\n').encode()
    contract = dict(schema_version=SCHEMA, statistics_sha256=hashlib.sha256(content).hexdigest(),
                    joint_delta_dims=list(JOINTS), absolute_gripper_dims=list(GRIPPERS),
                    state_encoding='absolute radians/metres',
                    action_encoding='joint delta to anchor state; grippers absolute metres',
                    horizon=horizon, window_start=window_start, anchor_stride=1,
                    training_episodes=episodes, state_rows=state_rows,
                    anchors=sum(n - 1 for n in counts), valid_action_rows=action_rows,
                    padding_included=False, quantile_method='linear', sources=sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(output.suffix + '.tmp').write_bytes(content)
    sidecar = contract_path(output)
    sidecar.with_suffix(sidecar.suffix + '.tmp').write_text(json.dumps(contract, indent=2) + '\n')
    output.with_suffix(output.suffix + '.tmp').replace(output)
    sidecar.with_suffix(sidecar.suffix + '.tmp').replace(sidecar)
    return statistics, contract


def load_delta_statistics(path, *, episodes, horizon=50, window_start=1, dataset=None):
    """Fail closed on absolute/raw, wrong-split, changed or wrong-window statistics."""
    import math

    path = Path(path)
    sidecar = contract_path(path)
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(f'missing delta statistics/contract: {path}; run action_statistics first')
    contract = json.loads(sidecar.read_text())
    expected = dict(schema_version=SCHEMA, joint_delta_dims=list(JOINTS),
                    absolute_gripper_dims=list(GRIPPERS), horizon=horizon,
                    window_start=window_start, anchor_stride=1, padding_included=False,
                    state_encoding='absolute radians/metres',
                    action_encoding='joint delta to anchor state; grippers absolute metres',
                    quantile_method='linear',
                    training_episodes=sorted(set(episodes)))
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f'delta statistics contract mismatch: {key}')
    if digest(path) != contract['statistics_sha256']:
        raise ValueError('delta statistics checksum mismatch')
    statistics = json.loads(path.read_text())
    for feature in ('proprio', 'action'):
        stats = statistics['private'][feature]
        for key in ('min', 'max', 'mean', 'std', 'q01', 'q99'):
            values = stats[key]
            if len(values) != 14 or not all(math.isfinite(v) for v in values):
                raise ValueError(f'invalid {feature}.{key}')
        if any(lo > hi for lo, hi in zip(stats['q01'], stats['q99'])):
            raise ValueError(f'invalid {feature} quantiles')
    if dataset is not None:
        root = Path(dataset)
        paths = sorted((root / 'data').rglob('*.parquet'))
        actual = sorted(int(p.stem.removeprefix('episode_')) for p in paths)
        if actual != expected['training_episodes']:
            raise ValueError('training dataset episodes differ from fitted split; refuse validation leakage')
        sources = {entry['path']: entry['sha256'] for entry in contract['sources']}
        actual_paths = {str(p.relative_to(root)) for p in paths}
        if set(sources) != actual_paths:
            raise ValueError('statistics source paths differ from training data')
        for path in paths:
            if digest(path) != sources[str(path.relative_to(root))]:
                raise ValueError(f'statistics source checksum mismatch: {path}')
    return statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--episodes', help='end-exclusive START:END, e.g. 0:105')
    group.add_argument('--episodes-file', type=Path,
                       help='JSON list of episodes, or a soma-train split_manifest.json (uses its train list)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--horizon', type=int, default=50)
    parser.add_argument('--scratch', type=Path)
    args = parser.parse_args()
    if args.episodes is not None:
        first, end = map(int, args.episodes.split(':'))
        episodes = range(first, end)
    else:
        loaded = json.loads(args.episodes_file.read_text())
        episodes = loaded['train'] if isinstance(loaded, dict) else loaded
    fit(args.dataset, episodes, args.output, horizon=args.horizon, scratch=args.scratch)


if __name__ == '__main__':
    main()
