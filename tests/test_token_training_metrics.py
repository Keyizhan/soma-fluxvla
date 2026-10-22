"""Analytic checks for pooled diagnostics versus the actual optimization mean."""
import json

import pytest
import torch

from fluxvla.training.subtask.train import Loss, diagnostic_metrics, training_metrics, validate


class AnalyticModel(torch.nn.Module):
    def __init__(self, token_weight=1., flow_weight=2.):
        super().__init__()
        self.value=torch.nn.Parameter(torch.tensor(1.))
        self.text_loss_mode='unified_token'
        self.token_loss_weight=token_weight
        self.hl_loss_weight=self.fast_loss_weight=0.
        self.action_loss_weight=flow_weight
        self.rtc_training_config=None
        self.high_calls=0

    def forward_high_level(self,images,ids,*args,**kwargs):
        self.high_calls+=1
        ns,nf,ss,sf=ids[0].to(self.value)
        ss=ss*self.value;sf=sf*self.value
        total=(ns+nf).clamp_min(1)
        return dict(loss_subtask=ss/ns.clamp_min(1),loss_action_ce=sf/nf.clamp_min(1),
            target_count=ns,action_target_count=nf,subtask_nll_sum=ss,fast_nll_sum=sf,
            loss_token=(ss+sf)/total,token_subtask_contribution=ss/total,
            token_fast_contribution=sf/total)

    def forward_low_level(self,*args,**kwargs):
        return dict(loss=4*self.value,action_squared_error_sum=12*self.value,
                    valid_action_count=self.value.new_tensor(3.))


def batch(counts_and_sums):
    return dict(images=torch.zeros(1,1),real=torch.ones(1,dtype=torch.bool),
        high_input_ids=torch.tensor([counts_and_sums]),high_valid_mask=None,
        high_ar_mask=None,high_labels=None,img_masks=None,high_subtask_region_mask=None,
        high_action_region_mask=None,low_input_ids=None,low_valid_mask=None,
        actions=None,action_masks=None)


def test_microbatch_optimization_and_pooled_metrics_have_distinct_scopes():
    model=AnalyticModel()
    loss=Loss(model,True)
    statistics=torch.zeros(13,dtype=torch.float64)
    for values in ([1,3,2,3],[9,1,27,4]):
        objective,_,row=loss(batch(values),torch.full((3,),.5),return_statistics=True)
        objective.backward()
        statistics+=row
    metrics=training_metrics(statistics)
    # Optimization averages 1.25 and 3.1 equally. Pooling sees 14 tokens.
    assert metrics['optimization_token_ce']==pytest.approx(2.175)
    assert metrics['token_ce']==pytest.approx(36/14)
    assert metrics['optimization_token_subtask_contribution']==pytest.approx(1.6)
    assert metrics['optimization_token_fast_contribution']==pytest.approx(.575)
    assert metrics['optimization_subtask_loss_fraction']==pytest.approx(1.6/2.175)
    assert metrics['subtask_loss_fraction']==pytest.approx(29/36)
    assert metrics['subtask_token_fraction']==pytest.approx(10/14)
    assert metrics['subtask_ce']==pytest.approx(2.9)
    assert metrics['fast_ce']==pytest.approx(1.75)
    assert metrics['optimization_loss']==pytest.approx(10.175)
    assert model.value.grad.item()==pytest.approx(10.175)
    assert metrics['optimization_weighted_token_term']+metrics['optimization_weighted_flow_term']==pytest.approx(metrics['optimization_loss'])
    json.dumps(metrics,allow_nan=False)


def test_rank_averaging_applies_only_to_optimization_statistics():
    loss=Loss(AnalyticModel(),True)
    _,_,a=loss(batch([1,3,2,3]),torch.ones(3),return_statistics=True)
    _,_,b=loss(batch([9,1,27,4]),torch.ones(3),return_statistics=True)
    metrics=training_metrics(a+b,world=2)
    assert metrics['token_count']==14
    assert metrics['token_ce']==pytest.approx(36/14)
    assert metrics['optimization_token_ce']==pytest.approx(2.175)
    assert metrics['optimization_loss']==pytest.approx(10.175)


def test_expert_training_marks_uncomputed_text_as_null():
    model=AnalyticModel(token_weight=0,flow_weight=1)
    objective,_,statistics=Loss(model,True)(batch([1,3,2,3]),torch.ones(3),return_statistics=True)
    assert model.high_calls==0
    metrics=training_metrics(statistics)
    assert objective.item()==4
    for key in ('subtask_ce','fast_ce','token_ce','subtask_loss_fraction',
                'optimization_token_ce','optimization_token_subtask_contribution',
                'optimization_subtask_loss_fraction'):
        assert metrics[key] is None
    assert metrics['optimization_weighted_token_term']==0
    assert metrics['text_metrics_computed'] is False
    json.dumps(metrics,allow_nan=False)


def test_zero_ce_has_defined_token_fractions_but_null_loss_fractions():
    metrics=diagnostic_metrics([0,0,0],[2,6,0])
    assert metrics['token_ce']==0
    assert metrics['subtask_token_fraction']==.25
    assert metrics['fast_token_fraction']==.75
    assert metrics['subtask_loss_fraction'] is None
    assert metrics['fast_loss_fraction'] is None
    assert metrics['flow_mse'] is None
    json.dumps(metrics,allow_nan=False)


def test_expert_validation_still_computes_real_token_diagnostics(monkeypatch):
    import fluxvla.training.subtask.train as module
    rows=[[1,3,2,3],[9,1,27,4]]
    monkeypatch.setattr(module,'make_batch',lambda indices,*args,**kwargs: batch(rows[indices[0]]))
    model=AnalyticModel(token_weight=0,flow_weight=1)
    metrics=validate(Loss(model,True),[0,1],None,None,dict(micro_batch=1),16,
                     torch.device('cpu'),4,1,0)
    assert model.high_calls==2
    assert metrics['subtask_ce']==pytest.approx(2.9)
    assert metrics['fast_ce']==pytest.approx(1.75)
    assert metrics['token_ce']==pytest.approx(36/14)
    assert metrics['subtask_token_count']==10
    assert metrics['fast_token_count']==4
    assert metrics['flow_mse']==4


def test_unified_objective_rejects_separate_region_accumulation_factors():
    with pytest.raises(ValueError,match='equal accumulation'):
        Loss(AnalyticModel(),True)(batch([1,3,2,3]),torch.tensor([1.,.5,1.]))
