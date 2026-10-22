"""Three-term loss tests: subtask CE + FAST CE + flow MSE, plus the leak guard."""
import copy
import runpy
from pathlib import Path

import pytest
import torch

from fluxvla import register_training_components
register_training_components()
from fluxvla.engines import build_vla_from_cfg

VOCAB = 64


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


@pytest.fixture
def model():
    torch.manual_seed(5)
    return build_vla_from_cfg(tiny_config()).eval()


def layout(fast_ids=(30, 31), batch=2):
    """[BOS prefix 2 tok] [subtask 2 tok + EOS] [action 1 tok + FAST ids + '|' + EOS]."""
    prefix = [2, 3, 4]
    subtask = [5, 6, 1]
    action = [7] + list(fast_ids) + [8, 1]
    ids = prefix + subtask + action
    n_prefix, n_subtask = len(prefix), len(subtask)
    subtask_region = [False] * n_prefix + [True] * n_subtask + [False] * len(action)
    action_region = [False] * (n_prefix + n_subtask) + [True] * len(action)
    loss = [a or b for a, b in zip(subtask_region, action_region)]
    out = dict(
        hl_input_ids=torch.tensor([ids] * batch),
        hl_valid_mask=torch.ones(batch, len(ids), dtype=torch.bool),
        hl_ar_mask=torch.tensor([loss] * batch),
        hl_labels=torch.tensor([[t if m else -100 for t, m in zip(ids, loss)]] * batch),
        hl_subtask_region_mask=torch.tensor([subtask_region] * batch),
        hl_action_region_mask=torch.tensor([action_region] * batch),
    )
    # First text token must stay conditioning context; the layout already satisfies it.
    return out


def image_batch(batch=2, generator=None):
    return torch.randn(batch, 9, 16, 16, generator=generator), \
        torch.tensor([[True, False, True]]).repeat(batch, 1)


def test_fast_ce_is_isolated_from_subtask_ce(model):
    generator = torch.Generator().manual_seed(3)
    images, img_masks = image_batch(generator=generator)
    a = layout()
    b = layout(fast_ids=(40, 41))          # only the FAST segment changes
    out_a = model.forward_high_level(images, a['hl_input_ids'], a['hl_valid_mask'],
                                     a['hl_ar_mask'], a['hl_labels'], img_masks,
                                     hl_subtask_region_mask=a['hl_subtask_region_mask'],
                                     hl_action_region_mask=a['hl_action_region_mask'])
    out_b = model.forward_high_level(images, b['hl_input_ids'], b['hl_valid_mask'],
                                     b['hl_ar_mask'], b['hl_labels'], img_masks,
                                     hl_subtask_region_mask=b['hl_subtask_region_mask'],
                                     hl_action_region_mask=b['hl_action_region_mask'])
    # Causal masking: the subtask loss cannot see the FAST segment.
    assert torch.equal(out_a['loss_subtask'], out_b['loss_subtask'])
    assert not torch.equal(out_a['loss_action_ce'], out_b['loss_action_ce'])
    assert out_a['target_count'].item() == out_b['target_count'].item() > 0
    assert out_a['action_target_count'].item() == out_b['action_target_count'].item() > 0


def test_regions_partition_the_supervised_set(model):
    generator = torch.Generator().manual_seed(4)
    images, img_masks = image_batch(generator=generator)
    a = layout()
    out = model.forward_high_level(images, a['hl_input_ids'], a['hl_valid_mask'],
                                   a['hl_ar_mask'], a['hl_labels'], img_masks,
                                   hl_subtask_region_mask=a['hl_subtask_region_mask'],
                                   hl_action_region_mask=a['hl_action_region_mask'])
    total = out['target_count'].item() + out['action_target_count'].item()
    assert total == int(a['hl_labels'].ne(-100).sum())
    none = model.forward_high_level(images, a['hl_input_ids'], a['hl_valid_mask'],
                                    a['hl_ar_mask'], a['hl_labels'], img_masks)
    assert none['loss_action_ce'] is None
    # Without region masks the CE is the count-weighted mean over every supervised token.
    expected = ((out['loss_subtask'] * out['target_count']
                 + out['loss_action_ce'] * out['action_target_count']) / total).item()
    assert abs(none['loss'].item() - expected) < 1e-5


def test_three_term_loss_weights(model):
    model.hl_loss_weight, model.fast_loss_weight, model.action_loss_weight = 2.0, 3.0, 4.0
    generator = torch.Generator().manual_seed(5)
    images, img_masks = image_batch(generator=generator)
    a = layout()
    states = torch.zeros(2, 32)
    actions = torch.randn(2, 4, 32, generator=generator)
    result = model(images=images, states=states, img_masks=img_masks,
                   ll_input_ids=torch.tensor([[2, 3, 4, 5]] * 2),
                   ll_valid_mask=torch.tensor([[True, True, True, False]] * 2),
                   actions=actions, action_masks=torch.ones(2, 4, 32, dtype=torch.bool),
                   noise=torch.randn(2, 4, 32, generator=generator),
                   time=torch.full((2,), 0.5), **a)
    expected = (result['loss_subtask'] + result['loss_fast']
                + result['loss_action']).item()
    assert abs(result['loss'].item() - expected) < 1e-5
    assert torch.equal(result['loss_hl'], result['loss_subtask'])
    assert result['loss_fast'] > 0


def test_flow_branch_never_sees_fast_tokens(model):
    generator = torch.Generator().manual_seed(6)
    images, img_masks = image_batch(generator=generator)
    states = torch.zeros(2, 32)
    actions = torch.randn(2, 4, 32, generator=generator)
    clean = dict(images=images, states=states, ll_input_ids=torch.tensor([[2, 3, 4, 5]] * 2),
                 ll_valid_mask=torch.tensor([[True, True, True, False]] * 2),
                 actions=actions, img_masks=img_masks,
                 noise=torch.randn(2, 4, 32, generator=generator), time=torch.full((2,), 0.5))
    a = model.forward_low_level(**clean)
    b = model.forward_low_level(**clean)
    assert torch.equal(a['predictions'], b['predictions'])


def test_low_level_prefix_rejects_relocated_fast_ids():
    # A vocabulary small enough to host a FAST tail: [52..59] is the relocated range.
    torch.manual_seed(7)
    guarded = build_vla_from_cfg(
        tiny_config(fast_vocab_size=8, fast_skip_tokens=4)).eval()
    ids = torch.tensor([[2, 3, 4, 5, 55]])
    with pytest.raises(ValueError, match='must not contain FAST action tokens'):
        guarded.forward_low_level(images=torch.randn(1, 9, 16, 16),
                                  states=torch.zeros(1, 32), ll_input_ids=ids,
                                  ll_valid_mask=torch.ones(1, 5, dtype=torch.bool),
                                  actions=torch.randn(1, 4, 32), img_masks=torch.ones(1, 3, dtype=torch.bool))
    ok = torch.tensor([[2, 3, 4, 5, 51]])       # just below the range
    guarded.forward_low_level(images=torch.randn(1, 9, 16, 16), states=torch.zeros(1, 32),
                              ll_input_ids=ok, ll_valid_mask=torch.ones(1, 5, dtype=torch.bool),
                              actions=torch.randn(1, 4, 32), img_masks=torch.ones(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match='must not contain FAST action tokens'):
        guarded.sample_actions_given_subtask(images=torch.randn(1, 9, 16, 16),
                                             states=torch.zeros(1, 32), ll_input_ids=ids,
                                             ll_valid_mask=torch.ones(1, 5, dtype=torch.bool),
                                             img_masks=torch.ones(1, 3, dtype=torch.bool), num_steps=2)


def test_region_mask_validation(model):
    generator = torch.Generator().manual_seed(8)
    images, img_masks = image_batch(generator=generator)
    a = layout()
    overlapping = a['hl_action_region_mask'] | a['hl_subtask_region_mask']
    with pytest.raises(ValueError, match='disjoint'):
        model.forward_high_level(images, a['hl_input_ids'], a['hl_valid_mask'], a['hl_ar_mask'],
                                 a['hl_labels'], img_masks,
                                 hl_subtask_region_mask=a['hl_subtask_region_mask'],
                                 hl_action_region_mask=overlapping)
    with pytest.raises(ValueError, match='align to token positions'):
        model.forward_high_level(images, a['hl_input_ids'], a['hl_valid_mask'], a['hl_ar_mask'],
                                 a['hl_labels'], img_masks,
                                 hl_subtask_region_mask=a['hl_subtask_region_mask'][:, :3],
                                 hl_action_region_mask=a['hl_action_region_mask'][:, :3])
    with pytest.raises(ValueError, match='Provide both region masks'):
        model.forward_high_level(images, a['hl_input_ids'], a['hl_valid_mask'], a['hl_ar_mask'],
                                 a['hl_labels'], img_masks,
                                 hl_subtask_region_mask=a['hl_subtask_region_mask'])


def test_fast_ce_reaches_the_backbone(model):
    generator = torch.Generator().manual_seed(9)
    images, img_masks = image_batch(generator=generator)
    a = layout()
    model.zero_grad(set_to_none=True)
    out = model.forward_high_level(images, a['hl_input_ids'], a['hl_valid_mask'],
                                   a['hl_ar_mask'], a['hl_labels'], img_masks,
                                   hl_subtask_region_mask=a['hl_subtask_region_mask'],
                                   hl_action_region_mask=a['hl_action_region_mask'])
    out['loss_action_ce'].backward()
    backbone = sum(float(p.grad.pow(2).sum()) for n, p in model.named_parameters()
                   if n.startswith('llm_backbone') and p.grad is not None)
    assert backbone > 0, 'FAST CE must train the shared VLM'


def test_time_sampling_is_a_real_beta(model):
    torch.manual_seed(0)
    assert model.exact_beta_time is True
    samples = model.sample_time(40000, torch.device('cpu'))
    # Beta(1.5, 1) has mean 0.6 and no mass below 0.001 after the rescale.
    assert abs(samples.mean().item() - 0.6) < 0.01
    assert samples.min() >= 0.001 and samples.max() <= 1.0
    assert abs(samples.std().item() - 0.2619) < 0.01   # sd = sqrt(1.5/(2.5^2*3.5))


def test_generation_accepts_an_explicit_ar_mask(model):
    generator = torch.Generator().manual_seed(11)
    images, img_masks = image_batch(generator=generator)
    b = layout()
    ids = b['hl_input_ids']
    kwargs = dict(images=images, hl_input_ids=ids, hl_valid_mask=b['hl_valid_mask'],
                  img_masks=img_masks, eos_token_id=-1, pad_token_id=0, max_new_tokens=2)
    open_mask = torch.zeros_like(ids, dtype=torch.bool)
    causal_mask = torch.ones_like(ids, dtype=torch.bool)
    a = model.generate_subtask(**kwargs, ar_mask=open_mask, use_cache=True)
    c = model.generate_subtask(**kwargs, ar_mask=causal_mask, use_cache=True)
    assert a['token_ids'].shape == c['token_ids'].shape
    with pytest.raises(ValueError, match='ar_mask must match'):
        model.generate_subtask(**{**kwargs, 'hl_input_ids': ids[:, :3]}, ar_mask=open_mask)


def test_invalid_fast_vocabulary_is_rejected():
    with pytest.raises(ValueError, match='FAST vocabulary does not fit'):
        build_vla_from_cfg(tiny_config(fast_vocab_size=4096, fast_skip_tokens=4))


def test_reference_low_prefix_is_causal_and_cache_matches_uncached():
    from fluxvla.engines.utils.model_utils import make_att_2d_masks
    m = build_vla_from_cfg(tiny_config(low_level_text_attention='causal')).eval()
    images, image_masks = image_batch(batch=1)
    ids = torch.tensor([[2,3,4,5]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    _, full_valid, ar = m.embed_prefix(images, ids, image_masks, valid)
    mask = make_att_2d_masks(full_valid, ar)
    offset = full_valid.shape[1]-ids.shape[1]
    assert not mask[0, offset, offset+1]
    assert mask[0, offset+1, offset]
    assert not mask[0, 0, offset]
    states = torch.zeros(1,32)
    actions = torch.randn(1,4,32)
    noise = torch.randn_like(actions)
    time = torch.tensor([.4])
    a = m.forward_low_level(images,states,ids,valid,actions,image_masks,
                           noise=noise,time=time,use_prefix_cache=True)
    b = m.forward_low_level(images,states,ids,valid,actions,image_masks,
                           noise=noise,time=time,use_prefix_cache=False)
    torch.testing.assert_close(a['predictions'], b['predictions'], rtol=2e-4,atol=2e-4)


@pytest.mark.parametrize('preset,insulated,text_trainable',[
    ('hybrid',False,True),('hybrid_insulated',True,True),('action_expert',True,False)])
def test_recipe_gradient_paths(preset, insulated, text_trainable):
    from fluxvla.training.subtask.train import configure_trainable_parameters
    m=build_vla_from_cfg(tiny_config(stop_gradient_flow_to_prefix=insulated,
        low_level_text_attention='causal')).eval()
    configure_trainable_parameters(m,preset)
    assert m.llm_backbone.embed_tokens.weight.requires_grad == text_trainable
    images, masks=image_batch(batch=1)
    ids=torch.tensor([[2,3,4]])
    valid=torch.ones_like(ids,dtype=torch.bool)
    result=m.forward_low_level(images,torch.zeros(1,32),ids,valid,torch.randn(1,4,32),masks)
    result['loss'].backward()
    backbone_grad=m.llm_backbone.embed_tokens.weight.grad
    assert bool(backbone_grad is not None and backbone_grad.abs().sum()>0)==(not insulated)
    assert m.action_out_proj.projector.weight.grad is not None


def test_expert_only_training_step_does_not_execute_text_loss(monkeypatch):
    from fluxvla.training.subtask.train import Loss, configure_trainable_parameters
    m=build_vla_from_cfg(tiny_config(hl_loss_weight=0.,fast_loss_weight=0.,
        action_loss_weight=1.,stop_gradient_flow_to_prefix=True,
        low_level_text_attention='causal'))
    configure_trainable_parameters(m,'action_expert')
    def forbidden(*args,**kwargs):
        raise AssertionError('expert-only stage executed text branch')
    monkeypatch.setattr(m,'forward_high_level',forbidden)
    images,masks=image_batch(batch=1)
    b=dict(images=images,img_masks=masks,real=torch.tensor([True]),
        low_input_ids=torch.tensor([[2,3,4]]),low_valid_mask=torch.ones(1,3,dtype=torch.bool),
        actions=torch.randn(1,4,32),action_masks=torch.ones(1,4,32))
    optimizer=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3)
    before=m.llm_backbone.embed_tokens.weight.detach().clone()
    loss,raw=Loss(m,True)(b,torch.ones(3))
    assert raw[0]==0 and raw[1]==0 and torch.isfinite(loss)
    loss.backward();optimizer.step()
    torch.testing.assert_close(before,m.llm_backbone.embed_tokens.weight,rtol=0,atol=0)
