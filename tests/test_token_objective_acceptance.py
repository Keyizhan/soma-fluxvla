"""Independent token-CE acceptance using unreduced logits as the oracle."""
from datetime import timedelta
from contextlib import nullcontext
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from fluxvla.engines import build_vla_from_cfg
from fluxvla.training.subtask.train import Loss, configure_trainable_parameters, training_metrics
from test_subtask_fast_model import tiny_config, image_batch


def unified_model():
    torch.manual_seed(723)
    model = build_vla_from_cfg(tiny_config(text_loss_mode='unified_token',
        token_loss_weight=1., hl_loss_weight=0., fast_loss_weight=0., action_loss_weight=0.,
        loss_reduction='local', low_level_prefix_source='text',
        low_level_text_attention='bidirectional', low_level_task_independent=True)).float().eval()
    configure_trainable_parameters(model, 'stage1_fast')
    return model


def unequal_batch():
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10],
                        [2, 3, 4, 11, 12, 13, 14, 0, 0],
                        [2, 3, 4, 15, 16, 17, 0, 0, 0]])
    valid = ids != 0
    sub = torch.zeros_like(valid)
    sub[0, 3] = True
    sub[1, 3:6] = True
    sub[2, 3:5] = True
    fast = valid.clone()
    fast[:, :3] = False
    fast &= ~sub
    targets = sub | fast
    images, masks = image_batch(batch=3, generator=torch.Generator().manual_seed(77))
    return dict(images=images, img_masks=masks, high_input_ids=ids,
                high_valid_mask=valid, high_ar_mask=targets,
                high_labels=torch.where(targets, ids, -100),
                high_subtask_region_mask=sub, high_action_region_mask=fast,
                real=torch.tensor([True, True, False]))


def high_forward(model, batch, sample_weight=None):
    return model.forward_high_level(batch['images'], batch['high_input_ids'],
        batch['high_valid_mask'], batch['high_ar_mask'], batch['high_labels'],
        batch['img_masks'], has_hl=batch['real'], sample_weight=sample_weight,
        return_logits=True, hl_subtask_region_mask=batch['high_subtask_region_mask'],
        hl_action_region_mask=batch['high_action_region_mask'])


def token_oracle(output, batch, sample_weight=None):
    target = batch['high_labels'][:, 1:]
    selected = (target != -100) & batch['real'][:, None]
    nll = F.cross_entropy(output['logits'][:, :-1][selected], target[selected], reduction='none')
    weights = (torch.ones_like(nll) if sample_weight is None else
               sample_weight[:, None].expand_as(target)[selected])
    return (nll * weights).sum() / weights.sum().clamp_min(1)


@pytest.mark.parametrize('weighted', [False, True])
def test_token_loss_and_gradient_match_direct_logits_ce(weighted):
    model = unified_model()
    batch = unequal_batch()
    weights = torch.tensor([1., 3., 9.]) if weighted else None
    output = high_forward(model, batch, weights)
    expected = token_oracle(output, batch, weights)
    torch.testing.assert_close(output['loss_token'], expected, rtol=1e-6, atol=1e-6)
    assert output['target_count'].item() == (10 if weighted else 4)
    assert output['action_target_count'].item() == (8 if weighted else 6)
    torch.testing.assert_close(output['token_subtask_contribution'] + output['token_fast_contribution'],
                               expected, rtol=1e-6, atol=1e-6)
    parameter = model.llm_backbone.embed_tokens.weight
    actual_gradient = torch.autograd.grad(output['loss_token'], parameter, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, parameter)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-5, atol=1e-6)


def test_masked_batch_has_finite_zero_loss_and_zero_gradient():
    model = unified_model()
    batch = unequal_batch()
    batch['real'].zero_()
    output = high_forward(model, batch)
    assert output['token_count'].item() == 0
    assert output['loss_token'].item() == 0
    output['loss_token'].backward()
    gradient = model.llm_backbone.embed_tokens.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and not gradient.any()


def test_unified_regions_cannot_silently_drop_a_supervised_target():
    model = unified_model()
    batch = unequal_batch()
    batch['high_action_region_mask'][0, 4] = False
    with pytest.raises(ValueError):
        high_forward(model, batch)


def select_row(batch, row, real=True):
    result = {key: value[row:row+1].clone() for key, value in batch.items()}
    result['real'].fill_(real)
    return result


def ddp_worker(rank, directory):
    """Actual two-rank reduction, including an empty final microbatch on rank 1."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    directory = Path(directory)
    dist.init_process_group('gloo', init_method=(directory / 'rendezvous').as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        model = unified_model()
        reference = unified_model()
        batch = unequal_batch()
        local_rows = [(0, True), (1, True)] if rank == 0 else [(1, True), (2, False)]
        wrapped = DistributedDataParallel(Loss(model, False))
        statistics = torch.zeros(21, dtype=torch.float64)
        for micro_index, (row, real) in enumerate(local_rows):
            with wrapped.no_sync() if micro_index == 0 else nullcontext():
                loss, _, measured = wrapped(select_row(batch, row, real), torch.full((3,), .5), return_statistics=True)
                loss.backward()
            statistics += measured
        dist.all_reduce(statistics)
        metrics = training_metrics(statistics, world=2)
        # Fixed current contract: each rank/microbatch has equal optimizer weight.
        # It is intentionally not CE over a pooled union of all tokens.
        oracle = torch.zeros(())
        region_sums = torch.zeros(2)
        region_counts = torch.zeros(2)
        optimization_contributions = torch.zeros(2)
        for row, real in [(0, True), (1, True), (1, True), (2, False)]:
            chunk = select_row(batch, row, real)
            output = high_forward(reference, chunk)
            oracle = oracle + token_oracle(output, chunk) / 4
            targets = chunk['high_labels'][:, 1:]
            active = (targets != -100) & chunk['real'][:, None]
            nll = F.cross_entropy(output['logits'][:, :-1][active], targets[active], reduction='none')
            for region, key in enumerate(['high_subtask_region_mask', 'high_action_region_mask']):
                selected = chunk[key][:, 1:][active]
                region_sums[region] += nll[selected].detach().sum()
                region_counts[region] += selected.sum()
                optimization_contributions[region] += nll[selected].detach().sum() / active.sum().clamp_min(1) / 4
        assert metrics['optimization_token_ce'] == pytest.approx(float(oracle.detach()), rel=1e-6)
        assert metrics['token_ce'] == pytest.approx(float(region_sums.sum()/region_counts.sum()), rel=1e-6)
        for i, name in enumerate(['subtask', 'fast']):
            assert metrics[f'{name}_token_count'] == float(region_counts[i])
            assert metrics[f'token_{name}_contribution'] == pytest.approx(float(region_sums[i]/region_counts.sum()), rel=1e-6)
            assert metrics[f'optimization_token_{name}_contribution'] == pytest.approx(float(optimization_contributions[i]), rel=1e-6)
            assert metrics[f'optimization_{name}_loss_fraction'] == pytest.approx(float(optimization_contributions[i]/optimization_contributions.sum()), rel=1e-6)
        oracle.backward()
        actual = model.llm_backbone.embed_tokens.weight.grad
        expected = reference.llm_backbone.embed_tokens.weight.grad
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        (directory / f'rank-{rank}.json').write_text(json.dumps(dict(
            rank=rank, max_gradient_error=float((actual-expected).abs().max()),
            empty_microbatch_included=True, metrics=metrics)), encoding='utf-8')
    finally:
        dist.destroy_process_group()


def test_real_two_rank_gradient_accumulation_matches_local_token_oracle(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(ddp_worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert len(list(tmp_path.glob('rank-*.json'))) == 2


@pytest.mark.parametrize('preset', ['hybrid', 'hybrid_insulated', 'action_expert'])
def test_unified_recipes_preserve_gradient_boundaries_and_joint_weight(preset):
    from fluxvla.training.presets.pi05_subtask_fast import PRESETS

    recipe = PRESETS[preset]['model']
    keys = ('text_loss_mode', 'token_loss_weight', 'hl_loss_weight', 'fast_loss_weight',
            'action_loss_weight', 'stop_gradient_flow_to_prefix', 'loss_reduction',
            'low_level_prefix_source', 'low_level_text_attention', 'low_level_task_independent')
    config = {key: recipe[key] for key in keys}
    config['action_loss_weight'] = 2.5 if preset != 'action_expert' else 1.
    model = build_vla_from_cfg(tiny_config(**config)).float().eval()
    configure_trainable_parameters(model, preset)
    batch = unequal_batch()
    batch.update(low_input_ids=batch['high_input_ids'][:, :3],
                 low_valid_mask=batch['high_valid_mask'][:, :3],
                 actions=torch.randn(3, 4, 32), action_masks=torch.ones(3, 4, 32, dtype=torch.bool),
                 flow_noise=torch.randn(3, 4, 32), flow_time=torch.full((3,), .4))
    total, _, statistics = Loss(model, True)(batch, torch.ones(3), return_statistics=True)
    high = high_forward(model, batch)
    low = model.forward_low_level(batch['images'], torch.zeros(3, 32), batch['low_input_ids'],
        batch['low_valid_mask'], batch['actions'], batch['img_masks'],
        action_masks=batch['action_masks'], has_action=batch['real'],
        noise=batch['flow_noise'], time=batch['flow_time'])
    expected = high['loss_token'] * model.token_loss_weight + low['loss'] * model.action_loss_weight
    torch.testing.assert_close(total, expected)
    low['loss'].backward()
    vlm = [p for name, p in model.named_parameters()
           if name.startswith(('llm_backbone', 'vision_backbone', 'projector'))]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in vlm) == (preset == 'hybrid')
    assert model.action_out_proj.projector.weight.grad is not None
    assert all(p.requires_grad for p in vlm) == (preset != 'action_expert')
    model.zero_grad(set_to_none=True)
    if preset != 'action_expert':
        high['loss_token'].backward()
        assert model.llm_backbone.embed_tokens.weight.grad.abs().sum() > 0
        assert model.action_out_proj.projector.weight.grad is None
    else:
        assert training_metrics(statistics)['text_metrics_computed'] is False
