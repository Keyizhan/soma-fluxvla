"""soma-train interop: split manifest format (D18) and ${VAR} config paths (D19)."""
import json

import pytest

from fluxvla.training.subtask.config import expand_env
from fluxvla.training.subtask.prepare import read_split_assignments


def test_soma_train_manifest(tmp_path):
    path = tmp_path / 'split.json'
    path.write_text(json.dumps(dict(schema_version='soma.train.split_manifest.v1',
                                    train=[0, 2], val=[1], test=[3])))
    assert read_split_assignments(path) == {0: 'train', 1: 'validation', 2: 'train', 3: 'test'}


def test_soma_train_manifest_overlap(tmp_path):
    path = tmp_path / 'split.json'
    path.write_text(json.dumps(dict(schema_version='soma.train.split_manifest.v1',
                                    train=[0], val=[0], test=[])))
    with pytest.raises(ValueError, match='two splits'):
        read_split_assignments(path)


def test_legacy_manifest(tmp_path):
    path = tmp_path / 'split.json'
    path.write_text(json.dumps(dict(episodes={'0': 'train', '1': 'validation'})))
    assert read_split_assignments(path) == {0: 'train', 1: 'validation'}


def test_expand_env(monkeypatch):
    monkeypatch.setenv('SOMA_SUBTASK_ROOT', '/w')
    assert expand_env({'a': ['${SOMA_SUBTASK_ROOT}/x', 3]}) == {'a': ['/w/x', 3]}
    monkeypatch.delenv('SOMA_SUBTASK_ROOT')
    with pytest.raises(ValueError, match='SOMA_SUBTASK_ROOT'):
        expand_env('${SOMA_SUBTASK_ROOT}/x')


def test_no_host_paths_in_subtask_configs():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / 'configs' / 'subtask'
    offenders = [p.name for p in root.rglob('*.json') if '/home/' in p.read_text()]
    assert not offenders, offenders
