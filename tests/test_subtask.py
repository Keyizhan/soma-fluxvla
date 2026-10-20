import copy
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fluxvla import register_training_components
register_training_components()
from fluxvla.engines import build_vla_from_cfg
from fluxvla.transforms.subtask_prompts import PrepareSubtaskPrompts


def tiny_config(kind='PI05Subtask'):
    cfg = runpy.run_path(str(Path(__file__).parents[1] / 'configs/pi05_piper.py'))['model']
    cfg.update(type=kind, pretrained_name_or_path=None, rtc_training_config=None,
               proj_width=16, n_action_steps=4)
    for key in ('llm_backbone', 'llm_expert'):
        cfg[key].update(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                        vocab_size=64, max_position_embeddings=512)
    cfg['llm_expert']['adarms_cond_dim'] = 16
    cfg['vision_backbone']['vision_config'].update(hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, image_size=16, patch_size=8)
    cfg['projector'].update(in_dim=16, out_dim=16)
    cfg['action_in_proj'].update(out_dim=16)
    cfg['action_out_proj'].update(in_dim=16)
    for key in ('time_mlp_in', 'time_mlp_out'):
        cfg[key].update(in_dim=16, out_dim=16)
    return cfg


@pytest.fixture
def model():
    torch.manual_seed(7)
    return build_vla_from_cfg(tiny_config()).eval()


def high_batch(batch=2):
    images = torch.randn(batch, 9, 16, 16)
    ids = torch.tensor([[2, 3, 4, 5, 6, 1, 0]]).repeat(batch, 1)
    valid = ids != 0
    ar = torch.tensor([[0, 0, 0, 1, 1, 1, 0]], dtype=torch.bool).repeat(batch, 1)
    labels = torch.where(ar, ids, -100)
    masks = torch.tensor([[True, False, True]]).repeat(batch, 1)
    return dict(images=images, hl_input_ids=ids, hl_valid_mask=valid,
                hl_ar_mask=ar, hl_labels=labels, img_masks=masks)


def test_future_isolation_shift_padding_and_projection(model):
    b = high_batch()
    out = model.forward_high_level(**b, return_logits=True)
    logits = out['logits']
    expected = F.cross_entropy(logits[:, :-1][b['hl_labels'][:, 1:] != -100],
                               b['hl_labels'][:, 1:][b['hl_labels'][:, 1:] != -100])
    torch.testing.assert_close(out['loss'], expected)
    changed = copy.deepcopy(b)
    changed['hl_input_ids'][:, 4] = 17
    changed_out = model.forward_high_level(**changed, return_logits=True)
    torch.testing.assert_close(logits[:, :4], changed_out['logits'][:, :4], rtol=0, atol=0)
    changed = copy.deepcopy(b)
    changed['hl_input_ids'][:, -1] = 42
    changed_out = model.forward_high_level(**changed, return_logits=True)
    torch.testing.assert_close(logits[:, :-1], changed_out['logits'][:, :-1], rtol=0, atol=0)
    hidden, _, offset = model.high_level_hidden(b['images'], b['hl_input_ids'],
        b['hl_valid_mask'], b['hl_ar_mask'], b['img_masks'])
    torch.testing.assert_close(logits, F.linear(hidden[:, offset:], model.llm_backbone.embed_tokens.weight).float())
    assert torch.isfinite(logits).all()


def test_zero_supervision_and_gradient(model):
    b = high_batch()
    b['hl_labels'].fill_(-100)
    out = model.forward_high_level(**b)
    assert out['loss'].item() == 0 and torch.isfinite(out['loss'])
    out['loss'].backward()
    model.zero_grad()
    b = high_batch()
    model.forward_high_level(**b)['loss'].backward()
    assert model.llm_backbone.embed_tokens.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.vision_backbone.parameters())
    assert all(p.grad is None for p in model.llm_expert.parameters())


def test_legacy_action_equivalence_and_low_level_gradient(model):
    base = build_vla_from_cfg(tiny_config('PI05FlowMatching')).eval()
    base.load_state_dict(model.state_dict(), strict=True)
    b = high_batch()
    states, actions = torch.randn(2, 32), torch.randn(2, 4, 32)
    noise, time = torch.randn_like(actions), torch.tensor([.3, .7])
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    old = base(images=b['images'], states=states, actions=actions, noise=noise, time=time,
        lang_tokens=b['hl_input_ids'], lang_masks=b['hl_valid_mask'], img_masks=b['img_masks'], action_masks=mask)
    new = model.forward_low_level(b['images'], states, b['hl_input_ids'], b['hl_valid_mask'],
        actions, b['img_masks'], action_masks=mask, noise=noise, time=time)
    torch.testing.assert_close(old['predictions'], new['predictions'], rtol=0, atol=0)
    torch.testing.assert_close(old['loss'], new['loss'])
    new['loss'].backward()
    assert model.action_out_proj.projector.weight.grad.abs().sum() > 0
    assert model.llm_backbone.embed_tokens.weight.grad.abs().sum() > 0


def test_last_valid_index_and_per_sample_eos(model, monkeypatch):
    b = high_batch()
    b['hl_valid_mask'] = b['hl_ar_mask'].logical_not() & b['hl_valid_mask']
    hidden, valid, _ = model.high_level_hidden(b['images'], b['hl_input_ids'],
        b['hl_valid_mask'], torch.zeros_like(b['hl_ar_mask']), b['img_masks'])
    physical_last = torch.where(valid, torch.arange(valid.shape[1])[None], -1).amax(-1)
    assert (physical_last != valid.sum(-1) - 1).all()
    recorded = []
    calls = 0
    def project(h):
        nonlocal calls
        recorded.append(h.clone())
        logits = torch.full((2, 64), -100.)
        logits[0, 1] = 100
        logits[1, 9 if calls == 0 else 1] = 100
        calls += 1
        return logits
    monkeypatch.setattr(model, 'project_text', project)
    out = model.generate_subtask(b['images'], b['hl_input_ids'], b['hl_valid_mask'],
        b['img_masks'], eos_token_id=1, pad_token_id=0, max_new_tokens=4)
    torch.testing.assert_close(recorded[0], hidden[torch.arange(2), physical_last])
    assert out['lengths'].tolist() == [1, 2]
    assert out['token_ids'].tolist() == [[1, 0, 0, 0], [9, 1, 0, 0]]
    assert out['finished'].all()


def test_global_task_not_in_low_level_api_and_checkpoint(model, tmp_path):
    b = high_batch()
    kwargs = dict(images=b['images'], states=torch.randn(2, 32), ll_input_ids=b['hl_input_ids'],
                  ll_valid_mask=b['hl_valid_mask'], img_masks=b['img_masks'], noise=torch.randn(2, 4, 32), num_steps=2)
    first = model.sample_actions_given_subtask(**kwargs)
    with pytest.raises(TypeError):
        model.sample_actions_given_subtask(**kwargs, overall_task='must not enter low-level')
    path = tmp_path / 'model.pt'
    torch.save(model.state_dict(), path)
    other = build_vla_from_cfg(tiny_config()).eval()
    other.load_state_dict(torch.load(path, weights_only=True), strict=True)
    torch.testing.assert_close(first, other.sample_actions_given_subtask(**kwargs), rtol=0, atol=0)
    assert list(model.state_dict()) == list(other.state_dict())


class FixtureTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 2, 1, 0
    def encode(self, text, add_special_tokens=False):
        return [ord(x) % 59 + 3 for x in text]
    def decode(self, ids):
        return str(ids)


def test_prompt_contract_and_overflow():
    p = PrepareSubtaskPrompts(FixtureTokenizer(), max_len=150, state_dim=2)
    out = p(dict(states=np.zeros(2), overall_task='overall', subtask='move'))
    ids, labels = out['hl_input_ids'], out['hl_labels']
    assert ids[0] == 2
    assert (labels[labels != -100] == np.array(FixtureTokenizer().encode('move') + [1])).all()
    assert (out['hl_ar_mask'] == (labels != -100)).all()
    assert np.array_equal(out['ll_input_ids'], p.low_level('move', np.zeros(2))['input_ids'])
    with pytest.raises(ValueError, match='no truncation'):
        p.high_level('overall' * 100, np.zeros(2), 'move')
    with pytest.raises(ValueError, match='Empty subtask'):
        p.high_level('overall', np.zeros(2), '')


def test_runtime_single_generation_goal_isolation_and_refresh(model, monkeypatch):
    from fluxvla.models.vlas.subtask_runtime import SubtaskRuntime
    # This test asserts the paper-form isolation (no overall task in the action context),
    # which is now the non-default switch.
    model.low_level_prefix_source = 'text'
    prompts = PrepareSubtaskPrompts(FixtureTokenizer(), max_len=300)
    monkeypatch.setattr(prompts, 'decode', lambda row: 'move')
    calls = []
    def generate(images, ids, valid, masks, **kwargs):
        calls.append((images.clone(), ids.clone()))
        return dict(token_ids=torch.tensor([[8, 1], [8, 1]]),
                    lengths=torch.tensor([2, 2]), finished=torch.tensor([True, True]))
    monkeypatch.setattr(model, 'generate_subtask', generate)
    runtime = SubtaskRuntime(model, prompts, num_steps=2)
    b = high_batch()
    inputs = dict(images=b['images'], states=torch.zeros(2, 32), img_masks=b['img_masks'],
                  noise=torch.randn(2, 4, 32))
    first = runtime.infer(**inputs, overall_tasks=['goal A']*2, observation_version=10)
    second = runtime.infer(**inputs, overall_tasks=['goal B']*2, observation_version=10)
    assert len(calls) == 2
    assert not torch.equal(calls[0][1], calls[1][1])
    torch.testing.assert_close(first['actions'], second['actions'], rtol=0, atol=0)
    assert first['subtasks'] == ['move', 'move']
    inputs['images'] = inputs['images'] + .1
    third = runtime.infer(**inputs, overall_tasks=['goal A']*2, observation_version=11)
    assert len(calls) == 3 and not torch.equal(calls[0][0], calls[2][0])
    assert third['trace']['observation_version'] == 11
    assert third['trace']['subtask_version'] == third['trace']['action_version'] == 3
    # Empty decoded text is still refused; EOS is no longer required because fast mode
    # trains "<subtask>;\nAction: <FAST>|<EOS>" and the runtime truncates at the delimiter.
    monkeypatch.setattr(model, 'generate_subtask', lambda *a, **kw:
        dict(token_ids=torch.tensor([[8], [8]]), finished=torch.tensor([False, True])))
    monkeypatch.setattr(prompts, 'decode', lambda row: '')
    with pytest.raises(ValueError, match='empty'):
        runtime.infer(**inputs, overall_tasks=['goal A']*2, observation_version=12)


def _distributed_reduction_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from fluxvla.models.vlas.pi05_subtask import global_weighted_mean
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        for empty_first_rank in (False, True):
            torch.manual_seed(21)
            net = DistributedDataParallel(torch.nn.Linear(2, 3, bias=False))
            reference = copy.deepcopy(net.module)
            x = torch.tensor([[.1, .3], [.7, -.2], [-.1, .8]])
            labels = torch.tensor([0, 2, 1])
            weights = torch.tensor([1., .5, 2.])
            split = 0 if empty_first_rank else 1
            subset = slice(0, split) if rank == 0 else slice(split, 3)
            nll = F.cross_entropy(net(x[subset]), labels[subset], reduction='none')
            loss = global_weighted_mean((nll*weights[subset]).sum(), weights[subset].sum())
            loss.backward()
            expected = (F.cross_entropy(reference(x), labels, reduction='none')*weights).sum()/weights.sum()
            expected.backward()
            torch.testing.assert_close(loss, expected)
            torch.testing.assert_close(net.module.weight.grad, reference.weight.grad)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_ddp_global_mean_with_unequal_counts_and_empty_rank(tmp_path):
    import torch.multiprocessing as mp
    mp.start_processes(_distributed_reduction_worker,
        args=(f'file://{tmp_path}/rendezvous',), nprocs=2, start_method='spawn', join=True)


def test_collator_combined_loss_and_optimizer_resume(model, tmp_path):
    from fluxvla.engines import build_collator_from_cfg
    config = runpy.run_path(str(Path(__file__).parents[1]/'configs/pi05_subtask_architecture.py'))
    prompts = PrepareSubtaskPrompts(FixtureTokenizer(), max_len=280)
    sample = prompts(dict(images=np.zeros((9,16,16),dtype=np.float32),
        img_masks=np.array([True,False,True]), states=np.zeros(32,dtype=np.float32),
        actions=np.ones((4,32),dtype=np.float32), action_masks=np.ones((4,),dtype=bool),
        overall_task='goal',subtask='move'))
    batch = build_collator_from_cfg(config['collator'])([sample])
    batch.update(noise=torch.zeros(1,4,32),time=torch.tensor([.5]))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
    output = model(**batch)
    torch.testing.assert_close(output['loss'],output['loss_hl']+output['loss_action'])
    output['loss'].backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    checkpoint=tmp_path/'resume.pt'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),checkpoint)
    restored=build_vla_from_cfg(tiny_config()).train()
    restored_optimizer=torch.optim.AdamW(restored.parameters(),lr=1e-4)
    state=torch.load(checkpoint,weights_only=True)
    restored.load_state_dict(state['model'],strict=True)
    restored_optimizer.load_state_dict(state['optimizer'])
    for network,opt in [(model,optimizer),(restored,restored_optimizer)]:
        network(**batch)['loss'].backward();opt.step()
    for a,b in zip(model.parameters(),restored.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_cached_flow_matches_joint_outputs_and_all_gradients(model):
    b = high_batch()
    kwargs = dict(images=b['images'], states=torch.randn(2,32),
                  ll_input_ids=b['hl_input_ids'], ll_valid_mask=b['hl_valid_mask'],
                  actions=torch.randn(2,4,32), img_masks=b['img_masks'],
                  noise=torch.randn(2,4,32), time=torch.tensor([.2,.8]),
                  action_masks=torch.tensor([[1,0,1,1],[1,1,1,0]]),
                  sample_weight=torch.tensor([.5,2.]))
    model.train()
    reference = model.forward_low_level(**kwargs, use_prefix_cache=False)
    reference['loss'].backward()
    grads = {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    cached = model.forward_low_level(**kwargs, use_prefix_cache=True)
    torch.testing.assert_close(reference['predictions'], cached['predictions'], rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(reference['loss'], cached['loss'])
    cached['loss'].backward()
    for n,p in model.named_parameters():
        if n in grads:
            # The split prefix does not consume final-layer prefix queries/MLP.
            # The joint attention path materializes structural zero grads there.
            if p.grad is None:
                assert not grads[n].any(),n
            else:
                torch.testing.assert_close(grads[n],p.grad,rtol=3e-4,atol=3e-6,msg=n)
        else:
            assert p.grad is None,n


def test_flow_insulation_blocks_prefix_but_preserves_ce_and_expert(model):
    b=high_batch()
    kwargs=dict(images=b['images'],states=torch.randn(2,32),
                ll_input_ids=b['hl_input_ids'],ll_valid_mask=b['hl_valid_mask'],
                actions=torch.randn(2,4,32),img_masks=b['img_masks'],
                noise=torch.randn(2,4,32),time=torch.tensor([.3,.7]))
    regular=model.forward_low_level(**kwargs,use_prefix_cache=True)
    model.stop_gradient_flow_to_prefix=True
    isolated=model.forward_low_level(**kwargs)
    torch.testing.assert_close(regular['predictions'],isolated['predictions'],rtol=0,atol=0)
    isolated['loss'].backward()
    for prefix in [model.vision_backbone,model.projector,model.llm_backbone]:
        assert all(p.grad is None or not p.grad.any() for p in prefix.parameters())
    assert model.action_out_proj.projector.weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.llm_expert.parameters())
    model.zero_grad(set_to_none=True)
    model.forward_high_level(**b)['loss'].backward()
    assert model.llm_backbone.embed_tokens.weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.vision_backbone.parameters())


def test_text_kv_logits_holes_padding_and_independent_eos(model, monkeypatch):
    b=high_batch()
    b['hl_valid_mask'] = torch.tensor([[1,1,1,0,0,0,0],[1,0,1,1,0,0,0]],dtype=torch.bool)
    kwargs=dict(images=b['images'],hl_input_ids=b['hl_input_ids'],
                hl_valid_mask=b['hl_valid_mask'],img_masks=b['img_masks'],
                eos_token_id=63,pad_token_id=0,max_new_tokens=5,return_logits=True)
    reference=model.generate_subtask(**kwargs,use_cache=False)
    cached=model.generate_subtask(**kwargs,use_cache=True)
    torch.testing.assert_close(reference['logits'],cached['logits'],rtol=2e-5,atol=2e-6)
    assert torch.equal(reference['token_ids'],cached['token_ids'])
    assert cached['cache_info']['length']<=cached['cache_info']['capacity']
    assert cached['cache_info']['capacity']-cached['cache_info']['length']<=4
    original_project=model.project_text
    for use_cache in [False,True]:
        calls=[]
        def project(hidden):
            calls.append(original_project(hidden))
            scores=torch.full((2,64),-100.)
            scores[0,1]=100
            scores[1,9 if len(calls)==1 else 1]=100
            return scores
        monkeypatch.setattr(model,'project_text',project)
        result=model.generate_subtask(**{**kwargs,'eos_token_id':1},use_cache=use_cache)
        assert result['lengths'].tolist()==[1,2]
        assert result['token_ids'].tolist()==[[1,0,0,0,0],[9,1,0,0,0]]
        assert result['finished'].all()


@pytest.mark.parametrize('horizon',[1,4])
def test_action_cache_is_read_only_and_request_local(model,horizon):
    model.n_action_steps=horizon
    b=high_batch()
    kwargs=dict(images=b['images'],states=torch.randn(2,32),
                ll_input_ids=b['hl_input_ids'],ll_valid_mask=b['hl_valid_mask'],
                img_masks=b['img_masks'],noise=torch.randn(2,horizon,32),num_steps=4)
    reference=model.sample_actions_given_subtask(**kwargs,use_cache=False)
    result=model.sample_actions_given_subtask(**kwargs,use_cache=True,return_trace=True)
    torch.testing.assert_close(reference,result['actions'],rtol=3e-5,atol=3e-6)
    assert len(set(result['cache_lengths']))==1
    assert result['cache_info']['length']==result['cache_info']['capacity']
    changed={**kwargs,'ll_input_ids':kwargs['ll_input_ids'].clone()}
    changed['ll_input_ids'][:,1]=18
    other=model.sample_actions_given_subtask(**changed,use_cache=True)
    assert not torch.equal(result['actions'],other)
    fresh=model.sample_actions_given_subtask(**kwargs,use_cache=True)
    torch.testing.assert_close(result['actions'],fresh,rtol=0,atol=0)


def test_text_cache_preserves_original_rope_precision_under_autocast(model,monkeypatch):
    b=high_batch()
    rotary=model.llm_backbone.rotary_emb.forward
    seen=[]
    def record(x,positions):
        seen.append(x.dtype)
        return rotary(x,positions)
    monkeypatch.setattr(model.llm_backbone.rotary_emb,'forward',record)
    kwargs=dict(images=b['images'],hl_input_ids=b['hl_input_ids'],
                hl_valid_mask=b['hl_valid_mask'],img_masks=b['img_masks'],
                eos_token_id=-1,pad_token_id=0,max_new_tokens=3,return_logits=True)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        a=model.generate_subtask(**kwargs,use_cache=False)
        original_dtypes=set(seen)
        seen.clear()
        c=model.generate_subtask(**kwargs,use_cache=True)
    assert original_dtypes==set(seen)=={torch.float32}
    torch.testing.assert_close(a['logits'],c['logits'],rtol=.03,atol=.003)
    assert torch.equal(a['token_ids'],c['token_ids'])


def test_explicit_fp32_text_decode_disables_outer_autocast(model):
    b=high_batch()
    model.text_decode_fp32=True
    kwargs=dict(images=b['images'],hl_input_ids=b['hl_input_ids'],hl_valid_mask=b['hl_valid_mask'],
                img_masks=b['img_masks'],eos_token_id=-1,pad_token_id=0,max_new_tokens=3,return_logits=True)
    a=model.generate_subtask(**kwargs,use_cache=False)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        c=model.generate_subtask(**kwargs,use_cache=True)
    torch.testing.assert_close(a['logits'],c['logits'],rtol=2e-5,atol=2e-6)
    assert torch.equal(a['token_ids'],c['token_ids'])


@pytest.mark.parametrize('stop_kind', ['limit', 'eos', 'sentinel'])
def test_returned_generation_cache_includes_final_emitted_token(model, monkeypatch, stop_kind):
    b=high_batch()
    images,ids,valid,masks=(b[k][:1] for k in ('images','hl_input_ids','hl_valid_mask','img_masks'))
    def project(hidden):
        scores=torch.full((1,64),-100.,device=hidden.device)
        scores[:,9]=100.
        return scores
    monkeypatch.setattr(model,'project_text',project)
    result=model.generate_subtask(images,ids,valid,masks,
        eos_token_id=9 if stop_kind=='eos' else 63,pad_token_id=0,
        max_new_tokens=2,use_cache=True,return_cache=True,
        stop_token_ids=[9] if stop_kind=='sentinel' else None)
    prefix=model.embed_prefix(images,ids,masks,valid)[0].shape[1]
    expected=prefix+int(result['lengths'][0])
    assert result['cache'].length == expected
    assert result['cache'].valid[0,expected-1]
    view=result['cache'].prefix(prefix)
    assert view.length == prefix and result['cache'].length == expected
