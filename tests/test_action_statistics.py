"""Regressions for fitting raw instead of anchor-relative action statistics."""
import json
from pathlib import Path
import runpy
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fluxvla.training.action_statistics import (
    JOINTS, contract_path, fit, load_delta_statistics, target_windows,
)


def episode(root, ep, *, extreme=False):
    n = 4
    state = np.arange(n * 14, dtype=np.float32).reshape(n, 14) * .01
    action = state + .1
    state[:, [6, 13]] = .04
    action[:, [6, 13]] = np.array([.01, .02, .03, .04])[:, None]
    if extreme:
        state += 1000
        action += 2000
    folder = root / 'data/chunk-000'
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        'observation.state': state.tolist(), 'action': action.tolist(),
        'frame_index': list(range(n)), 'episode_index': [ep] * n,
        'timestamp': [i / 30 for i in range(n)], 'overall_task': ['sort fabric'] * n,
    }), folder / f'episode_{ep:06d}.parquet')
    (root / 'meta').mkdir(exist_ok=True)
    (root / 'meta/info.json').write_text(json.dumps({
        'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        'chunks_size': 1000,
    }))
    return state, action


def test_exact_future_windows_keep_all_wrist_joints_and_absolute_grippers():
    states = np.full((4, 14), 7, dtype=np.float32)
    actions = np.arange(56, dtype=np.float32).reshape(4, 14)
    original = actions.copy()
    actual = np.concatenate(list(target_windows(states, actions, horizon=3)))
    expected = []
    for anchor in range(3):
        for future in range(anchor + 1, min(anchor + 4, len(actions))):
            row = actions[future].copy()
            row[list(JOINTS)] -= states[anchor, list(JOINTS)]
            expected.append(row)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actions, original)
    assert len(actual) == 6  # 3+2+1 valid rows, not 3*3 padded rows.
    np.testing.assert_array_equal(actual[0, [5, 12]], actions[1, [5, 12]] - 7)
    np.testing.assert_array_equal(actual[0, [6, 13]], actions[1, [6, 13]])


def test_fit_uses_only_training_and_excludes_padding(tmp_path):
    states, actions = episode(tmp_path, 0)
    episode(tmp_path, 105, extreme=True)
    output = tmp_path / 'stats.delta.json'
    stats, contract = fit(tmp_path, [0], output, horizon=3)
    targets = np.concatenate(list(target_windows(states, actions, horizon=3)))
    np.testing.assert_allclose(stats['private']['action']['mean'], targets.mean(0), atol=1e-7)
    np.testing.assert_allclose(stats['private']['action']['q01'],
                               np.quantile(targets, .01, axis=0))
    np.testing.assert_allclose(stats['private']['proprio']['mean'], states.mean(0), atol=1e-7)
    assert contract['anchors'] == 3 and contract['valid_action_rows'] == 6
    assert contract['training_episodes'] == [0] and not contract['padding_included']
    assert load_delta_statistics(output, episodes=[0], horizon=3) == stats
    with pytest.raises(ValueError, match='validation leakage'):
        load_delta_statistics(output, episodes=[0], horizon=3, dataset=tmp_path)
    with pytest.raises(FileExistsError):
        fit(tmp_path, [0], output, horizon=3)


def test_contract_rejects_raw_statistics_wrong_horizon_and_modification(tmp_path):
    episode(tmp_path, 0)
    output = tmp_path / 'stats.delta.json'
    stats, _ = fit(tmp_path, [0], output, horizon=3)
    with pytest.raises(ValueError, match='horizon'):
        load_delta_statistics(output, episodes=[0], horizon=50)
    with pytest.raises(ValueError, match='training_episodes'):
        load_delta_statistics(output, episodes=[105], horizon=3)
    assert load_delta_statistics(output, episodes=[0], horizon=3, dataset=tmp_path) == stats
    source = tmp_path / 'data/chunk-000/episode_000000.parquet'
    source.write_bytes(source.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='source checksum'):
        load_delta_statistics(output, episodes=[0], horizon=3, dataset=tmp_path)
    output.write_text(json.dumps(stats))  # Even a formatting edit invalidates the bound file.
    with pytest.raises(ValueError, match='checksum'):
        load_delta_statistics(output, episodes=[0], horizon=3)
    contract_path(output).unlink()
    with pytest.raises(ValueError, match='missing delta'):
        load_delta_statistics(output, episodes=[0], horizon=3)


def test_actual_training_transform_and_inverse_roundtrip(tmp_path):
    from fluxvla import register_training_components
    register_training_components()
    from fluxvla.transforms.normalize import NormalizeStatesAndActions
    from fluxvla.transforms.transform_actions import ProcessLiberoActions

    states, actions = episode(tmp_path, 0)
    statistics, _ = fit(tmp_path, [0], tmp_path / 'stats.json', horizon=3)
    stats = statistics['private']
    transform = ProcessLiberoActions(mask=[i not in (6, 13) for i in range(14)])
    normalize = NormalizeStatesAndActions(state_key='proprio', action_key='action',
                                         action_dim=32, state_dim=32, norm_type='quantile')
    physical = actions[1:].copy()
    sample = transform({'states': states[0].copy(), 'actions': physical.copy(), 'stats': stats})
    np.testing.assert_array_equal(sample['actions'],
                                  next(target_windows(states, actions, horizon=3))[:3])
    sample = normalize(sample)
    low, high = np.array(stats['action']['q01']), np.array(stats['action']['q99'])
    restored = (sample['actions'][:, :14] + 1) / 2 * (high - low + 1e-6) + low
    restored[:, JOINTS] += states[0, JOINTS]
    np.testing.assert_allclose(restored, physical, atol=1e-6)
