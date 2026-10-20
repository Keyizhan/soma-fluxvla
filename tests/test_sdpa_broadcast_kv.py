"""CUDA regression for broadcast KV heads and a final single-query tile."""
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from fluxvla.engines.utils.model_utils import (make_att_2d_masks,
                                               sdpa_attention_forward)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA SDPA')
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('length', [1025, 1026])
def test_broadcast_kv_last_query_matches_math(dtype, length):
    # A 768-token image prefix plus 256 padded text slots becomes 1025
    # positions after the first generated token. With one KV head, expand
    # creates zero head strides; the final single-query tile must retain its
    # own bias row rather than reading the image-only row at index zero.
    generator = torch.Generator(device='cuda').manual_seed(19)
    q = torch.randn(2, length, 8, 256, device='cuda', dtype=dtype,
                    generator=generator).transpose(1, 2)
    k = torch.randn(2, 1, length, 256, device='cuda', dtype=dtype,
                    generator=generator)
    v = torch.randn(2, 1, length, 256, device='cuda', dtype=dtype,
                    generator=generator)
    valid = torch.zeros(2, length, device='cuda', dtype=torch.bool)
    valid[:, :841] = True
    valid[:, 1024:] = True
    ar = torch.zeros_like(valid)
    ar[:, 768:841] = True
    ar[:, 1024:] = True
    allowed = make_att_2d_masks(valid, ar)[:, None]
    mask = torch.zeros_like(allowed, dtype=dtype)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    scale = 1 / math.sqrt(256)

    with sdpa_kernel(SDPBackend.MATH):
        reference = F.scaled_dot_product_attention(
            q[:, :, -1:].float(), k.float().expand(-1, 8, -1, -1),
            v.float().expand(-1, 8, -1, -1),
            attn_mask=mask[:, :, -1:].float(), scale=scale).transpose(1, 2)

    module = SimpleNamespace(num_key_value_groups=8, training=False)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        actual, _ = sdpa_attention_forward(module, q, k, v, mask, scale)

    tolerance = .002 if dtype == torch.bfloat16 else .00002
    torch.testing.assert_close(actual[:, -1:].float(), reference,
                               rtol=0, atol=tolerance)
