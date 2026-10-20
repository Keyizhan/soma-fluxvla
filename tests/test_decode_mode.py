"""Inference faithfulness: cached bf16 subtask decoding must warn, recompute must not."""
import runpy
from pathlib import Path

import pytest
import torch

from fluxvla import register_training_components
register_training_components()
from fluxvla.engines import build_vla_from_cfg


def model(**overrides):
    cfg = dict(runpy.run_path(str(Path(__file__).parents[1] / 'configs/pi05_piper.py'))['model'])
    cfg.update(type='PI05Subtask', pretrained_name_or_path=None, rtc_training_config=None,
               proj_width=16, n_action_steps=4, fast_vocab_size=8, fast_skip_tokens=4)
    for key in ('llm_backbone', 'llm_expert'):
        cfg[key].update(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                        vocab_size=64, max_position_embeddings=512)
    cfg['llm_expert']['adarms_cond_dim'] = 16
    cfg['vision_backbone']['vision_config'].update(hidden_size=16, intermediate_size=32,
                                                   num_hidden_layers=1, image_size=16, patch_size=8)
    cfg['projector'].update(in_dim=16, out_dim=16)
    cfg['action_in_proj'].update(out_dim=16)
    cfg['action_out_proj'].update(in_dim=16)
    for key in ('time_mlp_in', 'time_mlp_out'):
        cfg[key].update(in_dim=16, out_dim=16)
    cfg.update(**overrides)
    return build_vla_from_cfg(cfg).eval()


def kwargs():
    g = torch.Generator().manual_seed(5)
    return dict(images=torch.randn(2, 9, 16, 16, generator=g),
                hl_input_ids=torch.tensor([[2, 3, 4, 5, 6, 1, 0]] * 2),
                hl_valid_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 0]] * 2, dtype=torch.bool),
                img_masks=torch.tensor([[True, False, True]] * 2),
                eos_token_id=-1, pad_token_id=0, max_new_tokens=2)


def test_cached_bf16_decode_warns():
    # Only the bf16 attention policy is unfaithful; fp32 attention is the default now.
    m = model(attn_math='bf16')
    m.text_decode_fp32 = False
    with pytest.warns(RuntimeWarning, match='not numerically faithful'):
        m.generate_subtask(**kwargs(), use_cache=True)


def test_cached_decode_with_fp32_attention_does_not_warn():
    m = model()
    assert m.attn_math == 'fp32'
    m.text_decode_fp32 = False
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        m.generate_subtask(**kwargs(), use_cache=True)


def test_recompute_path_is_silent():
    m = model()
    m.text_decode_fp32 = False
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        m.generate_subtask(**kwargs(), use_cache=False)


def test_fp32_text_decode_is_accepted_as_the_faithful_cached_mode():
    m = model()
    m.text_decode_fp32 = True
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        m.generate_subtask(**kwargs(), use_cache=True)


def test_warning_can_be_disabled_explicitly():
    m = model(attn_math='bf16', warn_on_bf16_cache=False)
    m.text_decode_fp32 = False
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        m.generate_subtask(**kwargs(), use_cache=True)
