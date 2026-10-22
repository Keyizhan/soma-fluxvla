"""Ordinary and subtask training must use the same correct Beta law."""
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fluxvla.engines.utils.model_utils import sample_beta
from fluxvla.models.vlas.pi0_flowmatching import PI0FlowMatching
from fluxvla.models.vlas.pi05_subtask import PI05Subtask
from fluxvla.training.subtask.config import load_config, resolve_config
from fluxvla.training.subtask.randomness import flow_random


def test_matches_corrected_fluxvla_beta_branch():
    torch.manual_seed(31)
    expected = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.)).sample((4096,))
    torch.manual_seed(31)
    assert torch.equal(sample_beta(1.5, 1., 4096, 'cpu'), expected)
    model = SimpleNamespace(time_sampler='beta', time_beta_alpha=1.5, time_beta_beta=1.)
    torch.manual_seed(31)
    assert torch.equal(PI0FlowMatching.sample_time(model, 4096, 'cpu'), expected * .999 + .001)
    assert PI05Subtask.sample_time is PI0FlowMatching.sample_time


def test_sample_keyed_inverse_cdf_and_rng_isolation():
    actions = torch.zeros(10000, 1, 1)
    before = torch.get_rng_state()
    _, times = flow_random(list(range(10000)), actions, 1702, 'validation')
    assert torch.equal(before, torch.get_rng_state())
    # The legacy formula produces only ~6.67% above .9; correct Beta gives ~14.62%.
    assert abs(((times - .001) / .999 > .9).float().mean().item() - (1 - .9**1.5)) < .02


def test_configs_and_legacy_rejection():
    root = Path(__file__).resolve().parents[1]
    ordinary = runpy.run_path(str(root/'fluxvla/training/presets/pi05_piper.py'))['model']
    assert ordinary['time_sampler'] == 'beta'
    for name in ('fabric106', 'fabric106_hybrid_insulated', 'fabric106_two_stage'):
        assert load_config(root/f'configs/subtask/{name}.json')['time_sampler'] == 'beta'
    with pytest.raises(ValueError, match='time_sampler'):
        resolve_config({'time_sampler': 'legacy_power_ratio'})
    with pytest.raises(ValueError, match='time_sampler'):
        sample_beta(1.5, 1., 2, 'cpu', sampler='legacy_power_ratio')


def test_legacy_subtask_disable_flag_rejected():
    from test_subtask_fast_model import tiny_config
    from fluxvla.engines import build_vla_from_cfg
    with pytest.raises(ValueError, match='exact_beta_time=False'):
        build_vla_from_cfg(tiny_config(exact_beta_time=False))


@pytest.mark.parametrize('key', ['time_beta_alpha', 'time_beta_beta'])
@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True, '2'])
def test_invalid_beta_config(key, value):
    with pytest.raises(ValueError, match=key):
        resolve_config({key: value})


def test_general_keyed_beta_distribution_and_reordering():
    ids = list(range(6000))
    actions = torch.zeros(len(ids), 1, 1)
    state = torch.get_rng_state()
    noise, times = flow_random(ids, actions, 13, 'train', time_beta_alpha=2., time_beta_beta=3.)
    assert torch.equal(state, torch.get_rng_state())
    values = (times - .001) / .999
    assert abs(values.mean().item() - .4) < .015
    assert abs(values.var().item() - .04) < .005
    reverse_noise, reverse_times = flow_random(ids[::-1], actions, 13, 'train', time_beta_alpha=2., time_beta_beta=3.)
    assert torch.equal(times, reverse_times.flip(0))
    assert torch.equal(noise, reverse_noise.flip(0))
    default_noise, default_times = flow_random(ids, actions, 13, 'train')
    assert torch.equal(noise, default_noise)
    assert not torch.equal(times, default_times)


def test_batch_uses_configured_beta(monkeypatch):
    from fluxvla.training.subtask import train
    actions = torch.zeros(3, 1, 1)
    monkeypatch.setattr(train, 'collate', lambda *args: {'actions': actions})
    fx = SimpleNamespace(records=[{'sample_id': i} for i in range(3)])
    cfg = resolve_config({'seed': 17, 'time_beta_alpha': 2., 'time_beta_beta': 3.})
    batch = train.make_batch([0, 1, 2], fx, [{}, {}, {}], cfg, 16, 'cpu', 1, True)
    _, expected = flow_random([0, 1, 2], actions, 17, 'train', time_beta_alpha=2., time_beta_beta=3.)
    assert torch.equal(batch['flow_time'], expected)


def test_model_config_receives_beta():
    from fluxvla.training.subtask.fixtures import build_model
    model, cfg = build_model('hybrid', True, torch.device('cpu'), time_beta_alpha=2., time_beta_beta=3.)
    assert cfg['time_beta_alpha'] == model.time_beta_alpha == 2.
    assert cfg['time_beta_beta'] == model.time_beta_beta == 3.
