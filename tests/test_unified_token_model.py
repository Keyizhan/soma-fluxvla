"""Unified CE contract, diagnostic numerators, and explicit legacy mode."""
import pytest
import torch

from fluxvla.engines import build_vla_from_cfg
from test_subtask_fast_model import tiny_config, layout, image_batch


def unified(**kwargs):
    cfg = dict(text_loss_mode='unified_token', token_loss_weight=1.,
               hl_loss_weight=0., fast_loss_weight=0., loss_reduction='local')
    cfg.update(kwargs)
    return build_vla_from_cfg(tiny_config(**cfg)).eval()


def high(model, batch, **kwargs):
    images, masks = image_batch(batch=batch['hl_input_ids'].shape[0])
    return model.forward_high_level(images, batch['hl_input_ids'], batch['hl_valid_mask'],
        batch['hl_ar_mask'], batch['hl_labels'], masks,
        hl_subtask_region_mask=batch['hl_subtask_region_mask'],
        hl_action_region_mask=batch['hl_action_region_mask'], **kwargs)


def test_unified_regions_must_partition_all_supervised_tokens():
    m = unified()
    batch = layout(batch=1)
    batch['hl_action_region_mask'][0, -1] = False
    with pytest.raises(ValueError, match='cover exactly'):
        high(m, batch)
    # The legacy fallback retains its historical region semantics.
    old = build_vla_from_cfg(tiny_config()).eval()
    assert torch.isfinite(high(old, batch)['loss'])


@pytest.mark.parametrize('kwargs', [
    dict(hl_loss_weight=.15), dict(fast_loss_weight=.15),
    dict(text_loss_mode='separate'), dict(token_loss_weight=float('nan')),
    dict(token_loss_weight=-1), dict(action_loss_weight=float('inf')),
])
def test_ambiguous_or_invalid_loss_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        unified(**kwargs)


def test_weighted_diagnostics_use_the_same_effective_counts_and_nll():
    m = unified()
    out = high(m, layout(batch=2), sample_weight=torch.tensor([.25, 2.]))
    assert out['target_count'] == 3 * 2.25
    assert out['action_target_count'] == 5 * 2.25
    assert out['token_count'] == 8 * 2.25
    torch.testing.assert_close(out['loss_token'],
        (out['subtask_nll_sum'] + out['fast_nll_sum']) / out['token_count'])
    torch.testing.assert_close(out['loss_token'],
        out['token_subtask_contribution'] + out['token_fast_contribution'])
    torch.testing.assert_close(out['loss'], out['loss_token'])
    # A mean of region means would incorrectly give the two token groups equal mass.
    assert not torch.isclose(out['loss_token'], out['loss_subtask'] + out['loss_action_ce'])


def test_zero_effective_tokens_keeps_finite_zero_gradient_graph():
    m = unified()
    out = high(m, layout(batch=2), has_hl=torch.tensor([False, False]))
    for key in ('loss_token', 'subtask_nll_sum', 'fast_nll_sum',
                'token_subtask_contribution', 'token_fast_contribution'):
        assert out[key].item() == 0
        assert out[key].requires_grad
    out['loss_token'].backward()
    assert m.llm_backbone.embed_tokens.weight.grad is not None
    assert not m.llm_backbone.embed_tokens.weight.grad.any()
