"""RoPE precision policy: JAX (openpi/big_vision) semantics vs the legacy HF behaviour."""
import runpy
from pathlib import Path

import pytest
import torch

from fluxvla import register_training_components
register_training_components()
from fluxvla.engines import build_vla_from_cfg
from fluxvla.engines.utils.model_utils import apply_rotary_pos_emb, rotate_half

MAX_WAVELENGTH = 10_000.0


def angles(positions, head_dim):
    """Return (cos, sin) each shaped [B, L, head_dim//2], the JAX/big_vision layout."""
    exponents = (2.0 / head_dim) * torch.arange(head_dim // 2, dtype=torch.float32)
    timescale = MAX_WAVELENGTH ** exponents
    radians = positions.float()[..., None] / timescale[None, None, :]      # [B, L, D/2]
    return torch.cos(radians), torch.sin(radians)


def reference_jax_rope(x, cos, sin):
    """Port of openpi gemma_05._apply_rope: rotate in f32, downcast the RESULT."""
    dtype = x.dtype
    x = x.float()
    c, s = cos[:, None], sin[:, None]                                     # [B, 1, L, D/2]
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(dtype)


def batch(seq=5, heads=2, head_dim=8, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, heads, seq, head_dim, generator=g).to(dtype)
    positions = torch.arange(seq)[None].repeat(2, 1)
    return x, positions


def hf_layout(cos, sin):
    """HF rotary_emb returns cos/sin duplicated over head_dim."""
    return torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)


def test_jax_mode_matches_the_reference_formula_bitwise():
    x, positions = batch()
    cos, sin = hf_layout(*angles(positions, x.shape[-1]))
    got_q, got_k = apply_rotary_pos_emb(x, x, cos, sin, precision='jax')
    want = reference_jax_rope(x, cos[:, :, : cos.shape[-1] // 2],
                              sin[:, :, : sin.shape[-1] // 2])
    assert got_q.dtype == torch.bfloat16, 'jax mode must return the input dtype'
    assert torch.equal(got_q, want) and torch.equal(got_k, want)


def test_hf_mode_keeps_the_legacy_dtype_behaviour():
    x, positions = batch()
    cos, sin = hf_layout(*angles(positions, x.shape[-1]))
    got_bf16, _ = apply_rotary_pos_emb(x, x, cos.to(x.dtype), sin.to(x.dtype), precision='hf')
    assert got_bf16.dtype == x.dtype
    # Handed f32 cos/sin the legacy path silently upcasts instead: that is the
    # path-dependent dtype flip we removed by pinning the policy per model.
    got_f32, _ = apply_rotary_pos_emb(x, x, cos.float(), sin.float(), precision='hf')
    assert got_f32.dtype == torch.float32


def test_jax_mode_is_not_less_accurate_than_hf():
    x, positions = batch(dtype=torch.bfloat16)
    cos, sin = hf_layout(*angles(positions, x.shape[-1]))
    exact = reference_jax_rope(x, cos[:, :, : cos.shape[-1] // 2],
                               sin[:, :, : sin.shape[-1] // 2]).float()
    jax_q, _ = apply_rotary_pos_emb(x, x, cos, sin, precision='jax')
    hf_q, _ = apply_rotary_pos_emb(x, x, cos.to(x.dtype), sin.to(x.dtype), precision='hf')
    assert (jax_q.float() - exact).abs().max() <= (hf_q.float() - exact).abs().max()


def test_model_rejects_unknown_precision():
    cfg = dict(runpy.run_path(str(Path(__file__).parents[1] / 'configs/pi05_piper.py'))['model'])
    cfg.update(type='PI05Subtask', pretrained_name_or_path=None, rtc_training_config=None,
               rope_precision='nope')
    with pytest.raises(ValueError, match='rope_precision'):
        build_vla_from_cfg(cfg)


def test_default_precision_is_jax_for_the_subtask_model():
    cfg = dict(runpy.run_path(str(Path(__file__).parents[1] / 'configs/pi05_subtask_fast.py'))['model'])
    cfg['pretrained_name_or_path'] = None
    model = build_vla_from_cfg(cfg)
    assert model.rope_precision == 'jax'
