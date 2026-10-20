"""Attention-mask parity and fully-masked-row behaviour.

Reference implementation copied line-by-line from the JAX reference tree
``experiments/pi05_subtask_architecture/references/kewang`` at commit
``385d0e8330397ecfa29e9b675042b424c4d5386c``.

The task brief names ``src/openpi/models/gemma.py``; that file has no
``make_attn_mask`` (it only consumes a mask, see gemma.py:219-226).  The actual
mask constructor is ``make_attn_mask`` in ``src/openpi/models/pi05.py:21-46``
(identical copies also live in ``pi0.py:19`` and ``pi0_fast.py:23``).
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
from fluxvla.engines.utils.model_utils import make_att_2d_masks

VOCAB = 64


# ---------------------------------------------------------------------------
# 1. NumPy port of the JAX reference (pi05.py:21-46, commit 385d0e8)
# ---------------------------------------------------------------------------
def jax_make_attn_mask_numpy(input_mask, mask_ar):
    """Line-by-line NumPy port of ``openpi.models.pi05.make_attn_mask``.

    Source: references/kewang/src/openpi/models/pi05.py:21-46, commit
    385d0e8330397ecfa29e9b675042b424c4d5386c.  The body is unchanged except
    jnp -> np.
    """
    mask_ar = np.broadcast_to(mask_ar, input_mask.shape)
    cumsum = np.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return np.logical_and(attn_mask, valid_mask)


def _assert_mask_equal(pad, ar, context):
    pad_t = torch.as_tensor(pad, dtype=torch.bool)
    ar_t = torch.as_tensor(ar, dtype=torch.bool)
    got = make_att_2d_masks(pad_t, ar_t).numpy()
    want = jax_make_attn_mask_numpy(pad.astype(bool), ar.astype(bool))
    if np.array_equal(got, want):
        return
    diff = np.argwhere(got != want)[0]
    b, i, j = (int(x) for x in diff)
    raise AssertionError(
        f'{context}: mismatch at batch={b} query={i} key={j}\n'
        f'pad={pad[b].tolist()}\nar={ar[b].tolist()}\n'
        f'pytorch=        {got[b].astype(int).tolist()}\n'
        f'jax_numpy=      {want[b].astype(int).tolist()}')


def _all_binary_rows(length):
    codes = np.arange(1 << length, dtype=np.int64)
    return ((codes[:, None] >> np.arange(length)) & 1).astype(bool)


@pytest.mark.parametrize('length', [6, 7, 8, 9])
def test_exhaustive_all_pad_ar_pairs(length):
    """Exhaustively compare every (pad_row, ar_row) pair, literally batch=2.

    Sample 0 walks all ``2**L x 2**L`` pairs.  Sample 1 is the cyclic successor
    of both rows, so it is a different mask but still a legitimate batch mate.
    """
    rows = _all_binary_rows(length)
    n_rows = rows.shape[0]
    for pad_index in range(n_rows):
        pad = np.stack([rows[pad_index], rows[(pad_index + 1) % n_rows]])
        for ar_index in range(n_rows):
            ar = np.stack([rows[ar_index], rows[(ar_index + 1) % n_rows]])
            _assert_mask_equal(pad, ar, f'exhaustive length={length} '
                                        f'pad={pad_index} ar={ar_index}')


def _targeted_cases():
    return [
        # Pure causal / all-true AR.
        (np.ones((2, 6), bool), np.ones((2, 6), bool)),
        # Only the first token is an AR boundary.
        (np.ones((2, 6), bool), np.tile([1, 0, 0, 0, 0, 0], (2, 1)).astype(bool)),
        # Holes in the AR schedule (block-diagonal attention).
        (np.ones((2, 7), bool),
         np.array([[1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 0, 1]], bool)),
        # Tail padding, including one all-padding sample.
        (np.array([[1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], bool),
         np.array([[1, 0, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]], bool)),
        # Entirely false pad and AR rows.
        (np.zeros((2, 8), bool), np.zeros((2, 8), bool)),
        (np.zeros((2, 8), bool), np.ones((2, 8), bool)),
        # Interior invalid slots (e.g. a missing camera / padding hole).
        (np.array([[1, 1, 0, 1, 1, 1, 0], [1, 0, 1, 1, 0, 1, 1]], bool),
         np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0]], bool)),
        # AR all false while every input is valid.
        (np.ones((2, 9), bool), np.zeros((2, 9), bool)),
        # One empty sample next to a fully valid sample.
        (np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1]], bool),
         np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 0, 0, 1, 0, 0, 0, 1, 0]], bool)),
    ]


def test_targeted_cases():
    for pad, ar in _targeted_cases():
        _assert_mask_equal(pad, ar, 'targeted')


def test_random_cases():
    rng = np.random.default_rng(20261017)
    for trial in range(400):
        length = int(rng.integers(6, 10))
        pad = rng.random((2, length)) < rng.uniform(0.05, 0.95)
        ar = rng.random((2, length)) < rng.uniform(0.05, 0.85)
        # Sprinkle in the degenerate patterns from the targeted list.
        if trial % 17 == 0:
            pad[rng.integers(2)] = False
        if trial % 19 == 0:
            ar[:, 0] = True
        _assert_mask_equal(pad, ar, f'random trial={trial} length={length}')


def test_all_false_rows_come_only_from_padding():
    """A valid query always sees itself, so only pad=False can fully mask a row."""
    rng = np.random.default_rng(7)
    for _ in range(100):
        length = int(rng.integers(6, 10))
        pad = rng.random((2, length)) < 0.6
        ar = rng.random((2, length)) < 0.7
        mask = make_att_2d_masks(torch.from_numpy(pad), torch.from_numpy(ar))
        assert torch.equal(mask.any(-1), torch.from_numpy(pad))


# ---------------------------------------------------------------------------
# 2. Tiny model: fully-masked-row behaviour and cross-sample isolation
# ---------------------------------------------------------------------------
def tiny_config(**overrides):
    cfg = copy.deepcopy(runpy.run_path(
        str(Path(__file__).parents[1] / 'configs/pi05_piper.py'))['model'])
    cfg.update(type='PI05Subtask', pretrained_name_or_path=None, rtc_training_config=None,
               proj_width=16, n_action_steps=4)
    for key in ('llm_backbone', 'llm_expert'):
        cfg[key].update(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                        vocab_size=VOCAB, max_position_embeddings=512)
    cfg['llm_expert']['adarms_cond_dim'] = 16
    cfg['vision_backbone']['vision_config'].update(hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, image_size=16, patch_size=8)
    cfg['projector'].update(in_dim=16, out_dim=16)
    cfg['action_in_proj'].update(out_dim=16)
    cfg['action_out_proj'].update(in_dim=16)
    for key in ('time_mlp_in', 'time_mlp_out'):
        cfg[key].update(in_dim=16, out_dim=16)
    cfg.update(**overrides)
    return cfg


def image_batch(batch=2, generator=None):
    return torch.randn(batch, 9, 16, 16, generator=generator), \
        torch.tensor([[True, False, True]]).repeat(batch, 1)


@pytest.fixture(scope='module')
def tiny_models():
    torch.manual_seed(5)
    sdpa = build_vla_from_cfg(tiny_config(attention_implementation='sdpa')).eval()
    torch.manual_seed(5)
    eager = build_vla_from_cfg(tiny_config(attention_implementation='eager')).eval()
    # The HF backbone selects attention from config._attn_implementation, not
    # from the owning VLA's attribute. Set both so the full forward path really
    # exercises the requested backend.
    for model, implementation in ((sdpa, 'sdpa'), (eager, 'eager')):
        for module in (model.llm_backbone, model.llm_expert):
            module.config._attn_implementation = implementation
    assert all(torch.equal(a, b) for a, b in zip(sdpa.parameters(), eager.parameters()))
    return {'sdpa': sdpa, 'eager': eager}


def _layout(batch=2, length=5):
    ids = torch.tensor([[2, 3, 4, 5, 6]] * batch)
    valid = torch.ones(batch, length, dtype=torch.bool)
    ar = torch.tensor([[0, 1, 0, 1, 1]] * batch)
    return ids, valid, ar


@pytest.mark.parametrize('impl', ['sdpa', 'eager'])
def test_fully_masked_row_is_finite_uniform_attention(tiny_models, impl):
    model = tiny_models[impl]
    module = model.llm_backbone.layers[0].self_attn
    heads = module.config.num_attention_heads
    kv_heads = module.config.num_key_value_heads
    groups = module.num_key_value_groups
    torch.manual_seed(0)
    q = torch.randn(1, heads, 4, module.head_dim)
    k = torch.randn(1, kv_heads, 4, module.head_dim)
    v = torch.randn(1, kv_heads, 4, module.head_dim)

    # Build the exact additive mask the model would build for this row:
    # pad=[False, True, True, True], ar=all-zero => only query 0 is fully masked.
    pad = torch.tensor([[False, True, True, True]])
    ar = torch.zeros(1, 4, dtype=torch.bool)
    from_bool = model._prepare_attention_masks_4d(make_att_2d_masks(pad, ar))
    # pad[0]=False masks row 0 (query) and column 0 (key) identically.
    expected_mask = torch.zeros(1, 1, 4, 4, dtype=from_bool.dtype)
    expected_mask[0, 0, 0, :] = torch.finfo(from_bool.dtype).min
    expected_mask[0, 0, :, 0] = torch.finfo(from_bool.dtype).min
    assert torch.equal(from_bool, expected_mask)

    out, _ = model.attention_interface(module, q, k, v, from_bool, module.scaling)
    row = out[0, 0]  # (heads, head_dim), query 0 is fully masked
    expected_uniform = v[0].mean(dim=1).repeat_interleave(groups, dim=0)

    assert torch.isfinite(row).all(), 'fully masked row became NaN/Inf'
    assert not torch.all(row == 0), 'fully masked row silently became exactly zero'
    assert torch.allclose(row, expected_uniform, atol=1e-5, rtol=1e-5), \
        'fully masked row is not the uniform average of V'


@pytest.mark.parametrize('impl', ['sdpa', 'eager'])
def test_all_invalid_sample_does_not_change_other_sample_logits(tiny_models, impl):
    model = tiny_models[impl]
    images, img_masks = image_batch(generator=torch.Generator().manual_seed(3))
    ids, valid, ar = _layout()

    base_hidden, base_valid, _ = model.high_level_hidden(images, ids, valid, ar, img_masks)
    base_logits = model.project_text(base_hidden)

    bad_valid = valid.clone()
    bad_valid[0] = False
    bad_img_masks = img_masks.clone()
    bad_img_masks[0] = False
    bad_hidden, bad_full_valid, _ = model.high_level_hidden(
        images, ids, bad_valid, ar, bad_img_masks)
    bad_logits = model.project_text(bad_hidden)

    assert not bad_full_valid[0].any(), 'sample 0 was supposed to be fully invalid'
    assert torch.isfinite(bad_hidden[0]).all(), 'all-invalid sample produced NaN/Inf'
    assert not torch.all(bad_hidden[0] == 0), 'all-invalid sample silently became zero'
    # Bitwise equality, not allclose: per-sample attention must not mix samples.
    assert torch.equal(bad_hidden[1], base_hidden[1])
    assert torch.equal(bad_logits[1], base_logits[1])


@pytest.mark.parametrize('impl', ['sdpa', 'eager'])
def test_kv_cache_does_not_leak_between_samples(tiny_models, impl):
    model = tiny_models[impl]
    images, img_masks = image_batch(generator=torch.Generator().manual_seed(3))
    ids, valid, ar = _layout()

    states = torch.zeros(2, 32)
    x = torch.randn(2, 4, 32, generator=torch.Generator().manual_seed(7))
    time = torch.full((2,), 0.5)

    base_cache = model._prefill_low(images, ids, valid, img_masks)
    base_pred = model._cached_action_velocity(states, x, time, base_cache)

    bad_valid = valid.clone()
    bad_valid[0] = False
    bad_img_masks = img_masks.clone()
    bad_img_masks[0] = False
    bad_cache = model._prefill_low(images, ids, bad_valid, bad_img_masks)
    bad_pred = model._cached_action_velocity(states, x, time, bad_cache)

    for layer, ((base_k, base_v), (bad_k, bad_v)) in enumerate(
            zip(base_cache.layers, bad_cache.layers)):
        assert torch.equal(bad_k[1], base_k[1]), f'layer {layer} K leaked between samples'
        assert torch.equal(bad_v[1], base_v[1]), f'layer {layer} V leaked between samples'
        assert torch.isfinite(bad_k[0]).all() and torch.isfinite(bad_v[0]).all()
    assert torch.equal(bad_pred[1], base_pred[1])
    assert torch.isfinite(bad_pred[0]).all()
