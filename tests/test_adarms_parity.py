"""adaRMS parity: our GemmaRMSNorm must match openpi/big_vision gemma_05.RMSNorm.

Reference (gemma_05.py:112-131 @385d0e8):
    var   = mean(x.float()**2, -1, keepdim=True)          # f32
    normed= x * rsqrt(var + 1e-6)                          # f32
    if cond is None:  return (normed * (1 + scale)).astype(x.dtype), None
    modulation = Dense(width*3, kernel_init=zeros, dtype=x.dtype)(cond)
    scale, shift, gate = split(modulation[:, None], 3, -1)
    return (normed * (1 + scale) + shift).astype(x.dtype), gate
"""
import pytest
import torch

from fluxvla.models.backbones.llms.condition_gemma import GemmaRMSNorm


def reference_norm(x, cond, scale=None, kernel=None, bias=None, eps=1e-6):
    dtype = x.dtype
    var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    normed = x.float() * torch.rsqrt(var + eps)
    if cond is None:
        return (normed * (1.0 + scale.float())).to(dtype), None
    modulation = cond.float() @ kernel.float().T + bias.float()
    modulation = modulation[:, None, :]
    scale_m, shift_m, gate = torch.chunk(modulation, 3, dim=-1)
    return (normed * (1.0 + scale_m) + shift_m).to(dtype), gate.to(dtype)


def make(dim=8, cond_dim=4, seed=0):
    """adaRMS module (no `weight`, matching the reference, which has no scale param)."""
    torch.manual_seed(seed)
    module = GemmaRMSNorm(dim, eps=1e-6, cond_dim=cond_dim).eval()
    with torch.no_grad():
        module.dense.weight.uniform_(-0.3, 0.3)
        module.dense.bias.uniform_(-0.2, 0.2)
    return module


def make_plain(dim=8, seed=0):
    torch.manual_seed(seed)
    module = GemmaRMSNorm(dim, eps=1e-6, cond_dim=None).eval()
    with torch.no_grad():
        module.weight.uniform_(-0.5, 0.5)
    return module


def test_regular_branch_matches_reference():
    module = make_plain()
    x = torch.randn(2, 5, 8)
    got, gate = module(x, None)
    want, _ = reference_norm(x, None, scale=module.weight)
    assert gate is None
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-6)


def test_adaptive_branch_matches_reference():
    module = make()
    x = torch.randn(2, 5, 8)
    cond = torch.randn(2, 4)
    got, gate = module(x, cond)
    want, want_gate = reference_norm(x, cond, kernel=module.dense.weight,
                                     bias=module.dense.bias)
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-6)
    assert torch.allclose(gate, want_gate, atol=1e-6, rtol=1e-6)
    assert gate.shape == (2, 1, 8), 'gate is per-modulation, broadcast over tokens'


def test_bf16_output_dtype_is_preserved():
    x = torch.randn(2, 5, 8).to(torch.bfloat16)
    cond = torch.randn(2, 4)
    got, gate = make()(x, cond)
    assert got.dtype == torch.bfloat16 and gate.dtype == torch.bfloat16
    got_regular, _ = make_plain()(x, None)
    assert got_regular.dtype == torch.bfloat16


def test_fresh_initialisation_is_the_identity_modulation():
    """JAX zero-initialises BOTH the modulation kernel and its bias, so at init
    scale=shift=gate=0 and adaptive norm must equal plain RMSNorm."""
    module = GemmaRMSNorm(8, eps=1e-6, cond_dim=4).eval()
    assert torch.count_nonzero(module.dense.weight) == 0
    assert torch.count_nonzero(module.dense.bias) == 0, 'bias must be zero-initialised'
    plain_norm = GemmaRMSNorm(8, eps=1e-6, cond_dim=None).eval()
    assert torch.count_nonzero(plain_norm.weight) == 0
    x = torch.randn(2, 5, 8)
    cond = torch.randn(2, 4)
    adaptive, gate = module(x, cond)
    plain, _ = plain_norm(x, None)
    assert torch.allclose(adaptive, plain, atol=1e-6)
    assert torch.count_nonzero(gate) == 0


def test_adaptive_norm_without_cond_fails_loudly():
    module = make()
    with pytest.raises(AttributeError):
        module(torch.randn(1, 1, 8), None)


def test_eps_matches_the_reference_value():
    module = make_plain()
    assert module.eps == 1e-6
    x = torch.zeros(1, 1, 8)
    got, _ = module(x, None)
    assert torch.isfinite(got).all()
