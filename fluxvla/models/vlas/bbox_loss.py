"""Grouped bbox CE using existing LM logits, with rank-global reduction."""

import torch
import torch.distributed as dist

from fluxvla.transforms.bbox_grounding import PARTS


def update_denominators(packed, batches, device):
    """Count valid groups across a complete optimizer update, including all ranks."""
    import numpy as np

    counts = [0.0, 0.0, 0.0]
    for chunk in batches:
        for index in chunk:
            if index < 0:
                continue
            for j, name in enumerate(PARTS):
                ids = packed[index]["high"]["bbox_" + name + "_groups"]
                counts[j] += len(np.unique(ids[ids >= 0])) * (0.25 if name == "coordinate" else 1.0)
    result = torch.tensor(counts, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result)
    return result


def global_mean(total, count):
    if dist.is_available() and dist.is_initialized():
        denominator, value = count.detach().clone(), total.detach().clone()
        dist.all_reduce(denominator)
        dist.all_reduce(value)
        loss = total * dist.get_world_size() / denominator.clamp_min(1e-8)
        return loss + (value / denominator.clamp_min(1e-8) - loss.detach())
    return total / count.clamp_min(1e-8)


def bbox_statistics(nll, target_mask, groups, sample_weight=None, denominators=None):
    """Average tokens per group; coordinate groups have four equal parts per goal.

    Optional denominators cover an entire optimizer update (all microbatches and
    ranks). DDP then averages gradients; caller must not divide these terms again.
    """
    flat_loss = nll.new_zeros(target_mask.shape).masked_scatter(target_mask, nll)
    output = {}
    for index, name in enumerate(PARTS):
        ids = groups[name][:, 1:]
        valid = (ids >= 0) & target_mask
        weights = (
            torch.ones(ids.shape[0], device=nll.device) if sample_weight is None else sample_weight
        )
        # Batch row is part of each group identity. Vectorized scatter keeps the
        # reduction on-device rather than synchronizing once per coordinate.
        rows = torch.arange(ids.shape[0], device=ids.device)[:, None].expand_as(ids)
        pairs = torch.stack((rows[valid], ids[valid]), dim=1)
        unique, inverse = torch.unique(pairs, dim=0, return_inverse=True)
        sums = nll.new_zeros(len(unique)).scatter_add(0, inverse, flat_loss[valid])
        lengths = nll.new_zeros(len(unique)).scatter_add(
            0, inverse, torch.ones_like(inverse, dtype=nll.dtype)
        )
        group_weights = weights[unique[:, 0]] * (0.25 if name == "coordinate" else 1.0)
        total = (sums / lengths.clamp_min(1) * group_weights).sum() + nll.sum() * 0
        count = group_weights.sum()
        if denominators is None:
            loss = global_mean(total, count)
        else:
            world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
            loss = total * world / denominators[index].clamp_min(1e-8)
        output["bbox_" + name + "_sum"] = total
        output["bbox_" + name + "_count"] = count.detach()
        output["bbox_" + name + "_ce"] = loss
    return output
