"""The piper preset's statistics live in a JSON file, not in the module (T11)."""

import importlib
import json
from pathlib import Path

from fluxvla.training.presets import pi05_piper


def _table():
    return json.loads((Path(pi05_piper.__file__).with_name("pi05_piper_statistics.json")).read_text())


def test_preset_loads_the_committed_table():
    stats = pi05_piper.train_dataloader["dataset"]["dataset_statistics"]
    table = _table()
    assert stats == table
    assert len(table["private"]["action"]["q01"]) == 14
    assert len(table["private"]["proprio"]["q01"]) == 14


def test_environment_points_at_another_task(tmp_path, monkeypatch):
    table = _table()
    table["private"]["action"]["q01"] = [0.0] * 14  # a different task's table
    path = tmp_path / "other_task.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    monkeypatch.setenv("SOMA_PI05_STATS", str(path))
    try:
        reloaded = importlib.reload(pi05_piper)
        assert reloaded.train_dataloader["dataset"]["dataset_statistics"] == table
    finally:
        monkeypatch.delenv("SOMA_PI05_STATS")
        importlib.reload(pi05_piper)


def test_missing_table_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SOMA_PI05_STATS", str(tmp_path / "nope.json"))
    try:
        importlib.reload(pi05_piper)
    except FileNotFoundError as error:
        assert "SOMA_PI05_STATS" in str(error)
    else:
        raise AssertionError("a missing statistics file must fail, not fall back")
    finally:
        monkeypatch.delenv("SOMA_PI05_STATS")
        importlib.reload(pi05_piper)
