"""evaluate_subtask: absolute-joint MSE and the soma-train eval_report.v1 output (T4)."""
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
evaluate = runpy.run_path(str(ROOT / 'scripts/evaluate_subtask.py'))['evaluate']


class FakePolicy:
    camera_keys = ('a', 'b', 'c')
    device = torch.device('cpu')
    model = SimpleNamespace(n_action_steps=2, max_action_dim=32)
    config = SimpleNamespace(prefix_source='text', subtask_switch_confirmations=1, num_denoise_steps=10)
    prompts = object()

    def reset(self):
        pass

    def infer(self, payload, noise=None):
        actions = np.zeros((2, 14), np.float32)
        actions[:, 0] = 1.0     # 1 rad error on joint 0
        actions[:, 6] = 0.01    # 1 cm error on the left gripper
        return dict(actions=actions, subtask='reach', trace={})


def _inputs(n=3):
    arrays = dict(states=np.zeros((n, 14), np.float32), images=np.zeros((n, 9, 4, 4), np.uint8),
                  actions=np.zeros((n, 2, 14), np.float32))
    records = [dict(episode=5, frame=i, overall_task='t', subtask='reach') for i in range(n)]
    return arrays, records


def _split(tmp_path, val):
    path = tmp_path / 'split.json'
    path.write_text(json.dumps(dict(schema_version='soma.train.split_manifest.v1', policy='held_out',
                                    train=[0], val=val, test=[], split_digest='d' * 64)))
    return path


def test_absolute_mse_and_report(tmp_path):
    arrays, records = _inputs()
    summary = evaluate(FakePolicy(), arrays, records, tmp_path / 'out',
                       split_manifest=_split(tmp_path, [5]), release_id='r')
    assert summary['joint_mse_rad2'] == pytest.approx(1 / 12)
    assert summary['per_dim_mse'][0] == pytest.approx(1.0)
    assert summary['gripper_mse_m2'] == pytest.approx(1e-4 / 2)
    report = json.loads((tmp_path / 'out/eval_report.json').read_text())
    assert report['schema_version'] == 'soma.train.eval_report.v1'
    assert report['joints']['mse'] == pytest.approx(1 / 12) and report['episodes'] == [5]
    assert report['in_sample'] is False and report['split_digest'] == 'd' * 64


def test_episode_outside_split_is_refused(tmp_path):
    arrays, records = _inputs()
    with pytest.raises(ValueError, match='not in split'):
        evaluate(FakePolicy(), arrays, records, tmp_path / 'out', split_manifest=_split(tmp_path, [9]))
