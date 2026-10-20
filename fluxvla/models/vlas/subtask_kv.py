"""Request-local KV storage for subtask architecture probes.

AI implementation inspired by Ke-Wang1017/openpi_subtask at 385d0e8,
gemma_05.py. Physical storage offsets and logical RoPE positions are separate.
The caller explicitly selects append (text) or read-only (action) behavior;
sequence length never selects behavior. Training caches retain autograd graphs.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fluxvla.engines.utils.model_utils import apply_rotary_pos_emb, gated_residual


@dataclass
class PrefixKV:
    layers: list[tuple[torch.Tensor, torch.Tensor]]
    valid: torch.Tensor
    length: int

    @property
    def capacity(self) -> int:
        return self.valid.shape[1]

    def prefix(self, length: int) -> 'PrefixKV':
        """Read-only prefix view; preserve the complete generated cache."""
        if not 0 < length <= self.length:
            raise ValueError('prefix length outside generated cache')
        return PrefixKV([(k[:, :, :length], v[:, :, :length]) for k, v in self.layers],
                        self.valid[:, :length], length)

    def detached(self) -> 'PrefixKV':
        return PrefixKV([(k.detach(), v.detach()) for k, v in self.layers],
                        self.valid, self.length)

    def info(self) -> dict:
        return dict(length=self.length, capacity=self.capacity, layers=len(self.layers),
                    bytes=sum(x.numel() * x.element_size() for pair in self.layers for x in pair))


def run_kv_layers(owner, model, embeddings, positions, allowed, cond=None,
                  cache=None, append=False, capacity=None, valid=None, text_precision=False):
    """Run one expert with a fresh, appended, or read-only prefix KV.

    Attention masks use actual physical slots, including invalid camera/PAD holes.
    Fresh storage can reserve generation capacity. Appending is inference-only;
    read-only action calls concatenate transient action KV without modifying it.
    """
    if append and (cache is None or torch.is_grad_enabled()):
        raise ValueError('KV append requires an existing inference-only cache')
    query_len = embeddings.shape[1]
    if cache is None:
        if valid is None or valid.shape != embeddings.shape[:2]:
            raise ValueError('Prefill requires a matching validity mask')
        capacity = query_len if capacity is None else capacity
        if capacity < query_len:
            raise ValueError('KV capacity smaller than prefill')
        new_layers = []
    elif append and cache.length + query_len > cache.capacity:
        raise ValueError('KV capacity exhausted')
    if cache is not None and len(cache.layers) != len(model.layers):
        raise ValueError('KV layer count mismatch')
    attn_math = getattr(owner, 'attn_math', 'bf16')
    if attn_math not in ('fp32', 'bf16'):
        raise ValueError("attn_math must be 'fp32' or 'bf16'")
    mask4d = None
    hidden = embeddings
    for index, layer in enumerate(model.layers):
        residual = hidden
        normed, gate = layer.input_layernorm(hidden, cond=cond)
        shape = (*normed.shape[:-1], -1, layer.self_attn.head_dim)
        q = layer.self_attn.q_proj(normed).view(shape).transpose(1, 2)
        k = layer.self_attn.k_proj(normed).view(shape).transpose(1, 2)
        v = layer.self_attn.v_proj(normed).view(shape).transpose(1, 2)
        # RoPE precision policy must be identical on every path, otherwise the cached
        # and recomputed streams disagree under autocast (and the KV cache dtype flips).
        precision = getattr(owner, 'rope_precision', 'hf')
        if precision == 'jax':
            cos, sin = owner.llm_backbone.rotary_emb(embeddings.float(), positions)
        else:
            rotary_reference = embeddings if text_precision else q.transpose(1, 2)
            cos, sin = owner.llm_backbone.rotary_emb(rotary_reference, positions)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, precision=precision)
        if cache is None:
            new_layers.append((F.pad(k, (0, 0, 0, capacity-query_len)),
                               F.pad(v, (0, 0, 0, capacity-query_len))))
        elif append:
            stored_k, stored_v = cache.layers[index]
            stored_k[:, :, cache.length:cache.length+query_len].copy_(k)
            stored_v[:, :, cache.length:cache.length+query_len].copy_(v)
            k = stored_k[:, :, :cache.length+query_len]
            v = stored_v[:, :, :cache.length+query_len]
        else:
            stored_k, stored_v = cache.layers[index]
            k = torch.cat([stored_k[:, :, :cache.length], k], dim=2)
            v = torch.cat([stored_v[:, :, :cache.length], v], dim=2)
        if allowed.shape != (q.shape[0], query_len, k.shape[2]):
            raise ValueError('KV/query attention mask geometry mismatch')
        if mask4d is None:
            mask4d = owner._prepare_attention_masks_4d(allowed)
        if attn_math == 'fp32':
            # openpi computes the attention logits in float32 (preferred_element_type=f32)
            # while q/k/v stay in the model dtype; without this the single-query cached
            # step and the full recompute disagree by up to ~32 logits under bf16.
            attention, _ = owner.attention_interface(
                layer.self_attn, q.float(), k.float(), v.float(), mask4d.float(),
                layer.self_attn.scaling)
            attention = attention.to(v.dtype)
        else:
            attention, _ = owner.attention_interface(
                layer.self_attn, q, k, v, mask4d, layer.self_attn.scaling)
        attention = attention.flatten(2).to(layer.self_attn.o_proj.weight.dtype)
        hidden = gated_residual(residual, layer.self_attn.o_proj(attention), gate)
        residual = hidden
        normed, gate = layer.post_attention_layernorm(hidden, cond=cond)
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            normed = normed.to(torch.bfloat16)
        hidden = gated_residual(residual, layer.mlp(normed), gate)
    hidden, _ = model.norm(hidden, cond=cond)
    if cache is None:
        cache = PrefixKV(new_layers, F.pad(valid.bool(), (0, capacity-query_len)), query_len)
    elif append:
        if valid is None or valid.shape != embeddings.shape[:2]:
            raise ValueError('Append requires token validity')
        cache.valid[:, cache.length:cache.length+query_len] = valid
        cache.length += query_len
    return hidden, cache
