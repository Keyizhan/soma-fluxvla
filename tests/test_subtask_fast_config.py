"""The FAST presets must be importable and wired to the model's three-term loss."""
import runpy
from pathlib import Path

import pytest

from fluxvla import register_training_components
register_training_components()
from fluxvla.engines import build_vla_from_cfg

CONFIG = Path(__file__).parents[1] / 'configs/pi05_subtask_fast.py'


def load():
    return runpy.run_path(str(CONFIG))


def test_presets_use_unified_token_objectives():
    ns = load()
    weights = {name: preset['contract']['weights'] for name, preset in ns['PRESETS'].items()}
    assert weights['stage1_fast'] == dict(token=1.0, flow=0.0)
    assert weights['flow_only'] == dict(token=0.0, flow=1.0)
    assert weights['hybrid'] == weights['hybrid_insulated'] == dict(token=1.0, flow=1.0)
    # Default export is stage 1: the missing half of knowledge insulation.
    assert ns['model']['fast_loss_weight'] == ns['model']['hl_loss_weight'] == 0.0
    assert ns['model']['token_loss_weight'] == 1.0
    assert ns['model']['text_loss_mode'] == 'unified_token'
    assert ns['model']['action_loss_weight'] == 0.0


def test_contract_states_the_fast_invariants():
    contract = load()['architecture_contract']
    assert contract['state_dim'] == 14, 'state bins must not be padded with 18 zeros'
    assert contract['max_token_len'] == 256
    assert contract['fast_vocab_size'] == 2048
    assert contract['fast_skip_tokens'] == 128
    assert contract['low_level_sees_fast_tokens'] is False
    assert contract['fast_is_training_time_supervision_only'] is True


def test_collator_carries_the_region_masks():
    keys = load()['collator']['keys']
    for key in ('hl_subtask_region_mask', 'hl_action_region_mask',
                'hl_input_ids', 'll_input_ids', 'actions'):
        assert key in keys, key


def test_model_builds_with_the_fast_loss_enabled():
    model = build_vla_from_cfg(load()['PRESETS']['stage1_fast']['model'])
    assert model.hl_loss_weight == model.fast_loss_weight == 0.0
    assert model.token_loss_weight == 1.0
    assert model.text_loss_mode == 'unified_token'
    assert model.action_loss_weight == 0.0
    assert model.fast_vocab_size == 2048 and model.fast_skip_tokens == 128


def test_negative_weights_are_rejected():
    cfg = dict(load()['model'])
    cfg['fast_loss_weight'] = -1.0
    with pytest.raises(ValueError, match='fast_loss_weight'):
        build_vla_from_cfg(cfg)
