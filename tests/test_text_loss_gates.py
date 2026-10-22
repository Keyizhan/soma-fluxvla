import pytest
import torch
from fluxvla.training.subtask.text_loss import resolve_text_loss, weights_at, model_loss_settings
from fluxvla.training.subtask.config import resolve_config
from test_unified_token_model import unified, high
from test_subtask_fast_model import layout


def test_stage_schedule_and_resume_are_stateless():
    c = dict(mode='separate',weighting='linear',end_fast_weight=10.)
    assert weights_at(c,1,101)==(1.,1.)
    assert weights_at(c,51,101)==(1.,5.5)
    assert weights_at(c,101,101)==(1.,10.)
    assert weights_at(c,1000,101)==(1.,10.)
    assert weights_at(c,51,101)==weights_at(c,51,101)
    assert model_loss_settings(c,'action_expert')['fast_loss_weight']==0


@pytest.mark.parametrize('config', [dict(mode='typo'),dict(mode='unified_token',fast_weight=10),dict(mode='separate',weighting='linear',end_update=0),dict(mode='separate',fast_weight=float('nan')),dict(keyword_weight=3),dict(end_fast_weight=10)])
def test_invalid_configs(config):
    with pytest.raises(ValueError):resolve_text_loss(config)


@pytest.mark.parametrize('mode',['unified_token','separate'])
def test_split_objective_and_gradients_match_manual_formula(mode):
    m=unified(split_subtask_loss=True,subtask_key_weight=2.,subtask_template_weight=.1)
    m.text_loss_mode=mode
    batch=layout(batch=1)
    key=torch.zeros_like(batch['hl_subtask_region_mask']);key[:,3]=True
    template=batch['hl_subtask_region_mask'] & ~key
    out=high(m,batch,hl_subtask_key_mask=key,hl_subtask_template_mask=template)
    if mode=='unified_token':
        actual=out['loss_token_objective']
        expected=(2*out['subtask_key_nll_sum']+.1*out['subtask_template_nll_sum']+out['fast_nll_sum'])/out['token_count']
    else:
        actual=out['loss_subtask_objective']+10*out['loss_action_ce']
        expected=2*out['subtask_key_nll_sum']/out['subtask_key_count']+.1*out['subtask_template_nll_sum']/out['subtask_template_count']+10*out['fast_nll_sum']/out['action_target_count']
    torch.testing.assert_close(actual,expected)
    weight=m.llm_backbone.embed_tokens.weight
    g1=torch.autograd.grad(actual,weight,retain_graph=True)[0]
    g2=torch.autograd.grad(expected,weight)[0]
    torch.testing.assert_close(g1,g2)
    torch.testing.assert_close(out['loss_token'],(out['subtask_nll_sum']+out['fast_nll_sum'])/out['token_count'])


def test_split_masks_required_and_partitioned():
    m=unified(split_subtask_loss=True)
    batch=layout(batch=1)
    with pytest.raises(ValueError,match='requires'): high(m,batch)
    mask=batch['hl_subtask_region_mask']
    with pytest.raises(ValueError,match='partition'):high(m,batch,hl_subtask_key_mask=mask,hl_subtask_template_mask=mask)


def test_empty_template_has_finite_graph():
    m=unified(split_subtask_loss=True)
    batch=layout(batch=1);key=batch['hl_subtask_region_mask']
    out=high(m,batch,hl_subtask_key_mask=key,hl_subtask_template_mask=torch.zeros_like(key))
    assert out['subtask_template_ce']==0
    out['loss_token_objective'].backward()
    assert torch.isfinite(m.llm_backbone.embed_tokens.weight.grad).all()

def test_wrapper_uses_weighted_split_objective_but_reports_raw_metrics():
    from test_token_training_metrics import AnalyticModel, batch
    from fluxvla.training.subtask.train import Loss
    class SplitModel(AnalyticModel):
        def forward_high_level(self,*args,**kwargs):
            out=super().forward_high_level(*args,**kwargs)
            out['loss_subtask_objective']=7*self.value
            out['loss_token_objective']=11*self.value
            return out
    m=SplitModel(flow_weight=0.)
    objective,raw=Loss(m,False)(batch([1,3,2,3]),torch.ones(3))
    assert objective==11
    assert raw[0]==2 and raw[1]==1
    objective.backward();assert m.value.grad==11
    m.zero_grad();m.text_loss_mode='separate';m.hl_loss_weight=2;m.fast_loss_weight=10
    objective,_=Loss(m,False)(batch([1,3,2,3]),torch.ones(3))
    assert objective==24
    objective.backward();assert m.value.grad==24

def test_direct_model_forward_honors_split_masks_and_weights():
    from fluxvla.engines import build_vla_from_cfg
    from test_subtask_fast_model import tiny_config,image_batch
    m=build_vla_from_cfg(tiny_config(split_subtask_loss=True,subtask_key_weight=2.,subtask_template_weight=.1,hl_loss_weight=3.,fast_loss_weight=10.,action_loss_weight=0.)).eval()
    b=layout(batch=1);images,img_masks=image_batch(batch=1)
    key=torch.zeros_like(b['hl_subtask_region_mask']);key[:,3]=True
    b.update(hl_subtask_key_mask=key,hl_subtask_template_mask=b['hl_subtask_region_mask']&~key)
    h=m.forward_high_level(images,b['hl_input_ids'],b['hl_valid_mask'],b['hl_ar_mask'],b['hl_labels'],img_masks,
        **{k:v for k,v in b.items() if k.endswith('_mask') and k not in ('hl_valid_mask','hl_ar_mask')})
    out=m(images=images,states=torch.zeros(1,32),img_masks=img_masks,ll_input_ids=torch.tensor([[2,3,4]]),
        ll_valid_mask=torch.ones(1,3,dtype=torch.bool),actions=torch.zeros(1,4,32),noise=torch.zeros(1,4,32),time=torch.ones(1)*.5,**b)
    torch.testing.assert_close(out['loss'],3*h['loss_subtask_objective']+10*h['loss_action_ce'])
    torch.testing.assert_close(out['subtask_ce'],h['loss_subtask'])

def test_separate_actual_weighted_metrics_are_not_raw_token_fractions():
    from test_token_training_metrics import AnalyticModel,batch
    from fluxvla.training.subtask.train import Loss,training_metrics
    m=AnalyticModel(flow_weight=0.);m.text_loss_mode='separate';m.hl_loss_weight=1.;m.fast_loss_weight=10.
    _,_,stats=Loss(m,False)(batch([1,3,2,3]),torch.ones(3),return_statistics=True)
    row=training_metrics(stats)
    assert row['optimization_weighted_subtask_term']==2.
    assert row['optimization_weighted_fast_term']==10.
    assert row['optimization_weighted_fast_fraction']==pytest.approx(10/12)
    assert row['optimization_fast_loss_fraction']==pytest.approx(3/5)
