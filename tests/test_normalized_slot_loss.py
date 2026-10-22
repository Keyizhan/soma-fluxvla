import json
from pathlib import Path
import pytest
import torch
from fluxvla.training.subtask.text_loss import weights_at, model_loss_settings, resolve_text_loss
from fluxvla.training.subtask.config import load_config
from fluxvla.transforms.template_masks import TemplateMaskSpec, normalize
from test_unified_token_model import unified, high
from test_subtask_fast_model import layout


def test_normalized_schedule_and_scale_invariance():
    c=dict(mode='separate',weighting='linear',normalize_weights=True,end_fast_weight=10.)
    assert weights_at(c,1,101)==(.5,.5)
    assert weights_at(c,51,101)==pytest.approx((1/6.5,5.5/6.5))
    assert weights_at(c,101,101)==pytest.approx((1/11,10/11))
    a=model_loss_settings(dict(mode='separate',normalize_weights=True,subtask_weight=1,fast_weight=10,key_weight=1,template_weight=.1),'stage1_fast')
    b=model_loss_settings(dict(mode='separate',normalize_weights=True,subtask_weight=10,fast_weight=100,key_weight=10,template_weight=1),'stage1_fast')
    assert a==b
    assert a['subtask_key_weight']==pytest.approx(10/11)
    with pytest.raises(ValueError):resolve_text_loss(dict(normalize_weights='true'))


@pytest.mark.parametrize('mode',['separate','unified_token'])
def test_normalized_loss_and_gradient(mode):
    settings=model_loss_settings(dict(mode=mode,normalize_weights=True,split_subtask=True),'stage1_fast')
    m=unified(**settings)
    b=layout(batch=1);key=torch.zeros_like(b['hl_subtask_region_mask']);key[:,3]=True
    out=high(m,b,hl_subtask_key_mask=key,hl_subtask_template_mask=b['hl_subtask_region_mask'] & ~key)
    if mode=='separate':
        actual=m.hl_loss_weight*out['loss_subtask_objective']+m.fast_loss_weight*out['loss_action_ce']
        expected=.5*(out['subtask_key_ce']/1.1+out['subtask_template_ce']*.1/1.1)+.5*out['loss_action_ce']
    else:
        actual=out['loss_token_objective']
        expected=(out['subtask_key_nll_sum']+.1*out['subtask_template_nll_sum']+out['fast_nll_sum'])/(out['subtask_key_count']+.1*out['subtask_template_count']+out['action_target_count'])
    torch.testing.assert_close(actual,expected)
    w=m.llm_backbone.embed_tokens.weight
    torch.testing.assert_close(torch.autograd.grad(actual,w,retain_graph=True)[0],torch.autograd.grad(expected,w)[0])


def test_slots_cover_current_label_families_and_parallel_marker():
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/subtask/loss_examples/separate_split_fixed.json')
    spec=TemplateMaskSpec(templates=cfg['subtask_templates'])
    labels=[('Place fabric C in the far-left basket using the left arm.', ['place','c','far-left','left']),
            ('Flip the fabric piece on the staging mat face up.', ['flip','staging','face up']),
            ('Move a fabric piece from the pile to the staging mat.', ['move','pile','staging mat'])]
    for sentence,_ in list(labels):
        if not sentence.startswith('Flip'):
            labels.append((sentence+' || Move a fabric piece from the pile to the staging mat.', ['||','move','pile']))
    for label,keywords in labels:
        text=normalize(label)+';';mask=spec.key_char_mask(text)
        assert mask.any() and not mask.all()
        for word in keywords:
            start=text.index(word)
            assert mask[start:start+len(word)].all(), (label,word)
        # fabric is a format term in all these controlled label families.
        start=text.index('fabric');assert not mask[start:start+6].any()
