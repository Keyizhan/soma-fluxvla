import pytest
import torch

from fluxvla.training.subtask.common import lr_at
from fluxvla.optimizers.schedulers import get_cosine_schedule_with_warmup


@pytest.mark.parametrize('total', [7, 101])
def test_actual_learning_rates_match_ordinary_scheduler(total):
    p=torch.nn.Parameter(torch.ones(()))
    opt=torch.optim.AdamW([p],lr=1e-4)
    scheduler=get_cosine_schedule_with_warmup(opt,int(total*.05),total)
    # Ordinary policy explicitly starts at zero, including a zero-length warmup.
    opt.param_groups[0]['lr']=0.
    for update in range(1,total+1):
        assert lr_at(update,total,1e-4)==pytest.approx(opt.param_groups[0]['lr'])
        opt.step();scheduler.step()
    assert opt.param_groups[0]['lr']==0.


def test_weight_decay_groups_match_ordinary_policy():
    from fluxvla.training.subtask.train import build_optimizer
    model=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.LayerNorm(3))
    opt=build_optimizer(model,1e-4,.01)
    decay={id(p):g['weight_decay'] for g in opt.param_groups for p in g['params']}
    for name,p in model.named_parameters():
        assert decay[id(p)]==(.0 if p.ndim<=1 or name.endswith('.bias') else .01)


def test_rtc_cached_and_uncached_loss_and_mask():
    from test_subtask_fast_model import tiny_config,image_batch
    from fluxvla.engines import build_vla_from_cfg
    from fluxvla.engines.losses import reduce_action_bc_loss
    from fluxvla.engines.utils.rtc_training import apply_rtc_time_conditioning
    rtc=dict(enabled=True,max_delay=7,distribution='exponential',temperature=1.)
    model=build_vla_from_cfg(tiny_config(rtc_training_config=rtc,loss_reduction='local')).eval()
    images,img_masks=image_batch()
    actions=torch.randn(2,4,32);noise=torch.randn_like(actions)
    mask=torch.zeros_like(actions);mask[:,:,:14]=1;mask[1,2:]=0
    time=torch.tensor([.3,.7]);delays=torch.tensor([1,2])
    args=dict(images=images,states=torch.zeros(2,32),ll_input_ids=torch.tensor([[2,3],[2,3]]),
              ll_valid_mask=torch.ones(2,2,dtype=torch.bool),actions=actions,img_masks=img_masks,
              action_masks=mask,noise=noise,time=time,rtc_delays=delays)
    with torch.no_grad():
        cached=model.forward_low_level(**args,use_prefix_cache=True)
        uncached=model.forward_low_level(**args,use_prefix_cache=False)
    t,valid=apply_rtc_time_conditioning(time,mask,delays,4,clean_time=0.)
    assert t[0,0]==0 and t[1,:2].eq(0).all()
    assert cached['valid_action_count']==42
    expected=reduce_action_bc_loss((cached['predictions']-(noise-actions)[...,:14]).square(),valid[...,:14])
    torch.testing.assert_close(cached['loss'],expected)
    torch.testing.assert_close(cached['predictions'],uncached['predictions'],atol=2e-5,rtol=2e-5)


def test_all_presets_enable_ordinary_rtc_and_local_reduction():
    from fluxvla.training.presets.pi05_piper import model as ordinary
    from fluxvla.training.presets.pi05_subtask_fast import PRESETS
    for name in ('stage1_fast','hybrid','hybrid_insulated','action_expert'):
        assert PRESETS[name]['model']['rtc_training_config']==ordinary['rtc_training_config']
        assert PRESETS[name]['model']['loss_reduction']=='local'


@pytest.mark.parametrize('detach', [False, True])
def test_rtc_keeps_method_gradient_boundary(detach):
    from test_subtask_fast_model import tiny_config,image_batch
    from fluxvla.engines import build_vla_from_cfg
    model=build_vla_from_cfg(tiny_config(
        rtc_training_config=dict(enabled=True,max_delay=7),
        loss_reduction='local',flow_prefix_cache=True,
        stop_gradient_flow_to_prefix=detach))
    images,masks=image_batch()
    result=model.forward_low_level(images,torch.zeros(2,32),torch.tensor([[2,3],[2,3]]),
        torch.ones(2,2,dtype=torch.bool),torch.randn(2,4,32),masks,
        rtc_delays=torch.tensor([1,2]))
    result['loss'].backward()
    vlm=sum(p.grad.abs().sum().item() for p in model.llm_backbone.parameters() if p.grad is not None)
    expert=sum(p.grad.abs().sum().item() for p in model.llm_expert.parameters() if p.grad is not None)
    assert expert>0
    assert (vlm==0) if detach else (vlm>0)


def test_rtc_randomness_is_sample_keyed_and_does_not_consume_rng():
    from fluxvla.training.subtask.randomness import rtc_random
    cfg=dict(max_delay=7,distribution='exponential',temperature=1.)
    state=torch.get_rng_state()
    a=rtc_random(['a','b'],1,'train',2,cfg,'cpu')
    b=rtc_random(['b','a'],1,'train',2,cfg,'cpu')
    assert torch.equal(a,b.flip(0))
    assert torch.equal(state,torch.get_rng_state())


def test_local_reduction_matches_ordinary_with_unequal_masks(monkeypatch):
    from types import SimpleNamespace
    from fluxvla.models.vlas.pi05_subtask import PI05Subtask
    from fluxvla.engines.losses import reduce_action_bc_loss
    # Even in DDP, local reduction must not communicate element counts.
    monkeypatch.setattr(torch.distributed,'is_initialized',lambda: True)
    monkeypatch.setattr(torch.distributed,'all_reduce',lambda *a,**k: pytest.fail('unexpected global reduction'))
    prediction=torch.tensor([1.,3.,5.],requires_grad=True)
    local=[];ordinary=[]
    for indices in ([0],[1,2]):
        errors=prediction[indices].square()
        local.append(PI05Subtask._loss_mean(SimpleNamespace(loss_reduction='local'),errors.sum(),torch.tensor(float(len(indices)))))
        ordinary.append(reduce_action_bc_loss(errors,torch.ones_like(errors)))
    mean=sum(local)/2
    torch.testing.assert_close(mean,sum(ordinary)/2)
    assert not torch.isclose(mean,prediction.square().mean())
    torch.testing.assert_close(torch.autograd.grad(mean,prediction)[0],torch.tensor([1.,1.5,2.5]))
