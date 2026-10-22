"""Paper prefix information flow, independent low-level caches, and parity."""
import pytest
import torch
from test_subtask_fast_model import tiny_config, image_batch, layout
from fluxvla.engines import build_vla_from_cfg
from fluxvla.engines.utils.model_utils import make_att_2d_masks
from fluxvla.training.presets.pi05_subtask_fast import PRESETS


def model_for_test(**kwargs):
    torch.manual_seed(19)
    return build_vla_from_cfg(tiny_config(low_level_text_attention='bidirectional',
        low_level_prefix_source='text', low_level_task_independent=True, **kwargs)).eval()


def test_presets_use_paper_prefix_contract():
    for preset in PRESETS.values():
        cfg, contract = preset['model'], preset['contract']
        assert cfg['low_level_text_attention'] == 'bidirectional'
        assert cfg['low_level_task_independent'] is True
        assert contract['all_causal'] is False
        assert contract['low_level_sees_task'] is False
        assert contract['low_level_prompt_format'] == 'state_subtask'


def test_full_prefix_visibility_and_action_boundary(monkeypatch):
    m = model_for_test()
    images, masks = image_batch(batch=1)
    b = layout(batch=1)
    captured = []
    import fluxvla.models.vlas.pi05_subtask as module
    original = module.run_kv_layers
    def capture(*args, **kwargs):
        captured.append(args[4].clone())
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'run_kv_layers', capture)
    _, valid, offset = m.high_level_hidden(images, b['hl_input_ids'],
        b['hl_valid_mask'], b['hl_ar_mask'], masks)
    allowed = captured[-1]
    # A valid image attends all text conditions and vice versa, never outputs.
    assert allowed[0, 0, offset:offset+3].all()
    assert allowed[0, offset:offset+3, 0].all()
    assert not allowed[0, :offset+3, offset+3:].any()
    assert not allowed[0, offset+3, offset+4]
    assert allowed[0, offset+4, offset+3]
    assert not allowed[0, :, ~valid[0]].any()
    ids = torch.tensor([[2, 3, 4, 5, 0]])
    _, pv, pa = m.embed_prefix(images, ids, masks, ids != 0)
    _, sv, sa, _ = m.embed_suffix(torch.zeros(1, 32), torch.randn(1, 4, 32), torch.tensor([.4]))
    full = make_att_2d_masks(torch.cat([pv, sv], 1), torch.cat([pa, sa], 1))
    n = pv.shape[1]
    assert full[0, :n, :n][pv[0]][:, pv[0]].all()
    assert not full[0, :n, n:].any()
    assert full[0, n:, n:].all()
    assert full[0, n:, :n][:, pv[0]].all()


@pytest.mark.parametrize('future', [4, 8])
def test_future_subtask_and_fast_do_not_change_earlier_predictions(future):
    m = model_for_test()
    images, masks = image_batch(batch=1)
    b = layout(batch=1)
    ids = b['hl_input_ids']
    hidden, full_valid, offset = m.high_level_hidden(images, ids, b['hl_valid_mask'], b['hl_ar_mask'], masks)
    changed = ids.clone(); changed[:, future] = 42
    other, _, _ = m.high_level_hidden(images, changed, b['hl_valid_mask'], b['hl_ar_mask'], masks)
    # Invalid camera query slots are unused; only valid query representations are meaningful.
    keep = full_valid[:, :offset+future]
    torch.testing.assert_close(hidden[:, :offset+future][keep], other[:, :offset+future][keep], rtol=0, atol=0)
    # Conditioning is truly bidirectional, rather than merely blocking all text.
    changed = ids.clone(); changed[:, 2] = 43
    other, _, _ = m.high_level_hidden(images, changed, b['hl_valid_mask'], b['hl_ar_mask'], masks)
    assert not torch.equal(hidden[:, 0], other[:, 0])


@pytest.mark.parametrize('dtype,atol,rtol', [(torch.float32, 3e-5, 3e-5), (torch.bfloat16, .08, .03)])
def test_full_prefix_cached_text_and_action_parity_with_holes(dtype, atol, rtol):
    m = model_for_test(attn_math='bf16' if dtype == torch.bfloat16 else 'fp32').to(dtype)
    images, masks = image_batch(batch=2)
    images = images.to(dtype)
    ids = torch.tensor([[2, 3, 4, 0, 0], [2, 5, 6, 7, 0]])
    valid = ids != 0
    args = dict(images=images, hl_input_ids=ids, hl_valid_mask=valid, img_masks=masks,
        eos_token_id=-1, pad_token_id=0, max_new_tokens=4, ar_mask=torch.zeros_like(valid), return_logits=True)
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        cached = m.generate_subtask(**args, use_cache=True)
        full = m.generate_subtask(**args, use_cache=False)
        torch.testing.assert_close(cached['logits'], full['logits'], atol=atol, rtol=rtol)
        assert torch.equal(cached['token_ids'], full['token_ids'])
        states = torch.zeros(2, 32, dtype=dtype)
        actions, noise = torch.randn(2, 4, 32, dtype=dtype), torch.randn(2, 4, 32, dtype=dtype)
        kwargs = dict(images=images, states=states, ll_input_ids=ids, ll_valid_mask=valid,
            actions=actions, img_masks=masks, noise=noise, time=torch.tensor([.3, .7], dtype=dtype))
        cached = m.forward_low_level(**kwargs, use_prefix_cache=True)
        full = m.forward_low_level(**kwargs, use_prefix_cache=False)
        torch.testing.assert_close(cached['predictions'], full['predictions'], atol=atol, rtol=rtol)
        args = dict(images=images, states=states, ll_input_ids=ids, ll_valid_mask=valid,
            img_masks=masks, noise=noise, num_steps=3)
        torch.testing.assert_close(m.sample_actions_given_subtask(**args, use_cache=True),
            m.sample_actions_given_subtask(**args, use_cache=False), atol=atol, rtol=rtol)


def test_task_independent_model_rejects_external_generation_cache():
    m = model_for_test()
    images, masks = image_batch(batch=1)
    ids = torch.tensor([[2, 3, 4]])
    with pytest.raises(ValueError, match='rejects external prefix_cache'):
        m.sample_actions_given_subtask(images, torch.zeros(1, 32), ids, ids != 0,
            masks, prefix_cache=object())
    with pytest.raises(ValueError, match='requires rebuilt text prefix'):
        build_vla_from_cfg(tiny_config(low_level_task_independent=True, low_level_prefix_source='generation'))


@pytest.mark.parametrize('preset,insulated,trainable', [
    ('hybrid', False, True), ('hybrid_insulated', True, True), ('action_expert', True, False)])
def test_full_prefix_gradient_routes(preset, insulated, trainable):
    from fluxvla.training.subtask.train import configure_trainable_parameters
    m = model_for_test(stop_gradient_flow_to_prefix=insulated)
    configure_trainable_parameters(m, preset)
    images, masks = image_batch(batch=1)
    ids = torch.tensor([[2, 3, 4]])
    m.forward_low_level(images, torch.zeros(1, 32), ids, ids != 0,
        torch.randn(1, 4, 32), masks)['loss'].backward()
    for name in ('llm_backbone', 'vision_backbone', 'projector'):
        params = list(getattr(m, name).parameters())
        assert all(p.requires_grad == trainable for p in params)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params) == (not insulated)
    assert m.action_out_proj.projector.weight.grad.abs().sum() > 0
    if trainable:
        m.zero_grad(set_to_none=True)
        b = layout(batch=1)
        result = m.forward_high_level(images, b['hl_input_ids'], b['hl_valid_mask'], b['hl_ar_mask'],
            b['hl_labels'], masks, hl_subtask_region_mask=b['hl_subtask_region_mask'],
            hl_action_region_mask=b['hl_action_region_mask'])
        (result['loss_subtask'] + result['loss_action_ce']).backward()
        assert m.llm_backbone.embed_tokens.weight.grad.abs().sum() > 0
