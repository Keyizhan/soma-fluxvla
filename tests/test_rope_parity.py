"""Parity guards for RoPE positions and cached vs joint text decoding.

The reference JAX implementation uses ``cumsum(mask) - 1`` for every logical
position.  The cached paths use a prefix count plus a suffix cumsum, which is
algebraically the same expression.  These tests pin that identity, including
masks with interior camera/PAD holes, and check the text/action cache paths at
both FP32 and BF16 autocast precision.
"""
import copy
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

from fluxvla import register_training_components

register_training_components()
from fluxvla.engines import build_vla_from_cfg


def tiny_config():
    cfg = copy.deepcopy(
        runpy.run_path(str(Path(__file__).parents[1] / 'configs/pi05_piper.py'))['model'])
    cfg.update(type='PI05Subtask', pretrained_name_or_path=None, rtc_training_config=None,
               proj_width=16, n_action_steps=4)
    for key in ('llm_backbone', 'llm_expert'):
        cfg[key].update(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                        vocab_size=64, max_position_embeddings=512)
    cfg['llm_expert']['adarms_cond_dim'] = 16
    cfg['vision_backbone']['vision_config'].update(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, image_size=16, patch_size=8)
    cfg['projector'].update(in_dim=16, out_dim=16)
    cfg['action_in_proj'].update(out_dim=16)
    cfg['action_out_proj'].update(in_dim=16)
    for key in ('time_mlp_in', 'time_mlp_out'):
        cfg[key].update(in_dim=16, out_dim=16)
    return cfg


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(7)
    return build_vla_from_cfg(tiny_config()).eval()


def text_batch():
    # The second row has an interior invalid text slot, not just trailing PAD.
    ids = torch.tensor([[2, 3, 4, 5, 6, 1, 0], [2, 3, 4, 5, 6, 1, 0]])
    valid = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1, 0, 1, 1, 1, 0, 0]],
                         dtype=torch.bool)
    generator = torch.Generator().manual_seed(123)
    images = torch.randn(2, 9, 16, 16, generator=generator)
    img_masks = torch.tensor([[True, False, True]]).repeat(2, 1)
    return images, ids, valid, img_masks


def _rotary_capture(model, force_dtype):
    original = model.llm_backbone.rotary_emb.forward
    seen = []

    def wrapper(x, positions):
        if force_dtype is not None:
            x = x.to(force_dtype)
        seen.append(x.dtype)
        return original(x, positions)

    model.llm_backbone.rotary_emb.forward = wrapper
    return seen, original


def generate(model, cache, amp, force_dtype=None, max_new_tokens=6):
    images, ids, valid, img_masks = text_batch()
    seen, original = _rotary_capture(model, force_dtype)
    try:
        with torch.no_grad():
            context = (torch.autocast('cpu', dtype=torch.bfloat16)
                       if amp else torch.no_grad())
            with context:
                result = model.generate_subtask(
                    images, ids, valid, img_masks, eos_token_id=-1, pad_token_id=0,
                    max_new_tokens=max_new_tokens, use_cache=cache, return_logits=True)
    finally:
        model.llm_backbone.rotary_emb.forward = original
    return result, seen


def sample_actions(model, cache, amp):
    images, ids, valid, img_masks = text_batch()
    generator = torch.Generator().manual_seed(321)
    kwargs = dict(images=images, states=torch.randn(2, 32, generator=generator),
                  ll_input_ids=ids, ll_valid_mask=valid, img_masks=img_masks,
                  noise=torch.randn(2, 4, 32, generator=generator), num_steps=4)
    seen, original = _rotary_capture(model, None)
    try:
        with torch.no_grad():
            context = (torch.autocast('cpu', dtype=torch.bfloat16)
                       if amp else torch.no_grad())
            with context:
                result = model.sample_actions_given_subtask(**kwargs, use_cache=cache)
    finally:
        model.llm_backbone.rotary_emb.forward = original
    return result, seen


def test_position_formulas_agree_with_interior_holes():
    # JAX and our joint path both use cumsum(mask) - 1.  The cached suffix path
    # uses prefix.sum() + cumsum(suffix) - 1.  Prove the elementwise identity on
    # random masks that contain interior False entries.
    rng = np.random.default_rng(0)
    for prefix_len in range(1, 9):
        for _ in range(100):
            prefix = rng.random(prefix_len) < 0.5
            suffix = rng.random(7) < 0.5
            joint = np.cumsum(np.concatenate([prefix, suffix])) - 1
            cached = prefix.sum() + np.cumsum(suffix) - 1
            np.testing.assert_array_equal(joint[prefix_len:], cached)

            # Cached text decoding sums the combined validity mask for the newly
            # appended valid token; this is the same logical position.
            for step in range(len(suffix)):
                if suffix[step]:
                    combined = np.concatenate([prefix, suffix[:step]])
                    joint_append = np.cumsum(np.concatenate([combined, [True]]))[-1] - 1
                    cached_append = combined.sum() + 1 - 1
                    assert joint_append == cached_append

    # Concrete hole-heavy example used in the report.
    prefix = np.array([1, 1, 1, 0, 0, 0, 0], dtype=bool)
    suffix = np.array([1, 0, 1, 1, 1, 0, 0], dtype=bool)
    joint = np.cumsum(np.concatenate([prefix, suffix])) - 1
    assert joint[len(prefix):].tolist() == [3, 3, 4, 5, 6, 6, 6]
    assert (prefix.sum() + np.cumsum(suffix) - 1).tolist() == [3, 3, 4, 5, 6, 6, 6]


def test_fp32_cached_text_logits_match_joint(model):
    joint, _ = generate(model, cache=False, amp=False)
    cached, _ = generate(model, cache=True, amp=False)
    torch.testing.assert_close(joint['logits'], cached['logits'], rtol=2e-5, atol=2e-6)
    assert torch.equal(joint['token_ids'], cached['token_ids'])


def test_bf16_cached_text_parity_and_unified_rotary_dtype(model):
    joint, joint_dtypes = generate(model, cache=False, amp=True)
    cached, cached_dtypes = generate(model, cache=True, amp=True)
    # With FP32 parameters, the text backbone feeds RoPE its FP32 input hidden
    # state on the joint path and the cached text path deliberately does the same.
    assert joint_dtypes and cached_dtypes
    assert set(joint_dtypes) == set(cached_dtypes) == {torch.float32}
    torch.testing.assert_close(joint['logits'], cached['logits'], rtol=.03, atol=.003)
    assert torch.equal(joint['token_ids'], cached['token_ids'])


@pytest.mark.parametrize('force_dtype', [torch.float32, torch.bfloat16])
def test_forced_rotary_dtype_keeps_bf16_cache_parity(model, force_dtype):
    joint, joint_dtypes = generate(model, cache=False, amp=True, force_dtype=force_dtype)
    cached, cached_dtypes = generate(model, cache=True, amp=True, force_dtype=force_dtype)
    assert set(joint_dtypes) == set(cached_dtypes) == {force_dtype}
    torch.testing.assert_close(joint['logits'], cached['logits'], rtol=.03, atol=.003)
    assert torch.equal(joint['token_ids'], cached['token_ids'])


def test_bf16_cached_action_matches_joint_and_uses_same_rotary_dtype(model):
    joint, joint_dtypes = sample_actions(model, cache=False, amp=True)
    cached, cached_dtypes = sample_actions(model, cache=True, amp=True)
    # rope_precision='jax' feeds float32 into the rotary module on BOTH paths so that
    # cos/sin stay float32; the rotated q/k are then downcast back to the model dtype.
    # What matters is that the policy no longer depends on which path runs.
    assert joint_dtypes and cached_dtypes
    assert set(joint_dtypes) == set(cached_dtypes) == {torch.float32}
    torch.testing.assert_close(joint, cached, rtol=3e-5, atol=3e-6)
