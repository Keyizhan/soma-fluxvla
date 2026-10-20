"""Deployment must use the precision policy training used."""
import runpy
from pathlib import Path

from fluxvla import register_training_components
register_training_components()
from fluxvla.engines import build_vla_from_cfg


def preset(name):
    ns = runpy.run_path(str(Path(__file__).parents[1] / 'configs/pi05_subtask_fast.py'))
    return dict(ns['PRESETS'][name]['model'])


def test_presets_decode_text_in_the_training_precision():
    for name in ('stage1_fast', 'flow_only', 'hybrid', 'hybrid_insulated'):
        cfg = preset(name)
        assert cfg['text_decode_fp32'] is False, (
            f'{name}: fp32 text decoding is a different function than the bf16 training '
            'forward (measured 3x-26x larger deviation), keep it False')
        assert cfg['rope_precision'] == 'jax'
        assert cfg['attn_math'] == 'fp32'


def test_model_built_from_a_preset_keeps_the_policy():
    cfg = preset('stage1_fast')
    cfg['pretrained_name_or_path'] = None
    model = build_vla_from_cfg(cfg)
    assert model.attn_math == 'fp32' and model.rope_precision == 'jax'
    assert model.text_decode_fp32 is False


def test_contract_documents_the_policy():
    ns = runpy.run_path(str(Path(__file__).parents[1] / 'configs/pi05_subtask_fast.py'))
    contract = ns['architecture_contract']
    assert contract['rope_precision'] == 'jax'
    assert contract['attn_math'] == 'fp32'
