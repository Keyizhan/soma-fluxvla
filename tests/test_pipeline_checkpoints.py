"""Real serialization, strict deployment loads, and destructive-action gates."""
import json
from pathlib import Path
import sys

import pytest
import torch

from fluxvla.training.subtask import checkpoints as cp


@pytest.fixture
def fixture(tmp_path):
    cache=tmp_path/'cache';cache.mkdir()
    (cache/'samples.json').write_text(json.dumps({'normalization':{'state':{'q01':[0]*14,'q99':[1]*14},'action':{'q01':[0]*14,'q99':[1]*14}}}))
    tok=tmp_path/'tokenizer';tok.mkdir();(tok/'tokenizer.json').write_text('{}')
    model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Linear(4,2)).to(torch.bfloat16)
    model[0].requires_grad_(False)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    x=torch.ones(2,3,dtype=torch.bfloat16);model(x).sum().backward();opt.step()
    cfg=dict(cache=str(cache),tokenizer=str(tok),max_prompt_length=256,
             checkpoint=dict(format='full-state-deploy-v2',optimizer_save='final_only',save_every_steps=2,safety_reserve_gib=0))
    native=dict(type='PI05Subtask',n_action_steps=50,pretrained_name_or_path='/missing/base',
                vision_backbone=dict(vision_config=dict(image_size=224)))
    meta=dict(stage='stage2',preset='hybrid',update=2,total_updates=4,epoch=1,epoch_index=0,
              batch_cursor=2,cache_identity='abc',smoke=True)
    return tmp_path,model,opt,cfg,native,meta,x


def test_complete_frozen_state_and_strict_deployment(fixture):
    out,model,opt,cfg,native,meta,x=fixture
    full=cp.atomic_full(model,opt,out/'step-00000002.pt',meta,[{'rank':0}],0)
    state=cp.verify_full(full)
    assert set(state['model_state'])==set(model.state_dict())
    assert state['optimizer']['state']
    bundle=cp.export_bundle(state['model_state'],out/'step-00000002.deploy',cfg,native,meta)
    restored=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Linear(4,2)).to(torch.bfloat16)
    cp.load_bundle(restored,bundle)
    torch.testing.assert_close(model(x),restored(x),rtol=0,atol=0)
    assert json.loads((bundle/'model_config.json').read_text())['pretrained_name_or_path'] is None
    assert not list(bundle.rglob('*.pt'))
    assert not any('optimizer' in p.name for p in bundle.rglob('*'))
    # A deployment is the gate: no external archive receipt is needed.
    cp.remove_full_after_bundle(full,bundle)
    assert not full.exists()


def test_failed_new_save_preserves_previous(fixture,monkeypatch):
    out,model,opt,cfg,native,meta,x=fixture
    old=cp.atomic_full(model,opt,out/'step-00000002.pt',meta,[{}],0)
    old_sha=cp.digest(old)
    def fail(*args,**kwargs):raise OSError('injected disk write failure')
    monkeypatch.setattr(cp.torch,'save',fail)
    with pytest.raises(OSError):cp.atomic_full(model,opt,out/'step-00000004.pt',dict(meta,update=4),[{}],0)
    assert cp.digest(old)==old_sha
    cp.verify_full(old)


def test_failed_conversion_or_checksum_preserves_optimizer(fixture,monkeypatch):
    out,model,opt,cfg,native,meta,x=fixture
    old=cp.atomic_full(model,opt,out/'step-00000002.pt',meta,[{}],0)
    new=cp.atomic_full(model,opt,out/'step-00000004.pt',dict(meta,update=4),[{}],0)
    real_save=cp.save_file
    def fail(*args,**kwargs):raise OSError('injected deployment failure')
    monkeypatch.setattr(cp,'save_file',fail)
    with pytest.raises(OSError):cp.retire_previous(out,new,cfg,native,'stage2',0)
    assert old.exists() and cp.verify_full(old)['optimizer']['state']
    monkeypatch.setattr(cp,'save_file',real_save)
    bundle=cp.export_bundle(cp.verify_full(old)['model_state'],old.with_suffix('.deploy'),cfg,native,meta)
    with (bundle/'model.safetensors').open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError):cp.remove_full_after_bundle(old,bundle)
    assert old.exists()


def test_previous_full_becomes_minimal_and_latest_retains_optimizer(fixture):
    out,model,opt,cfg,native,meta,x=fixture
    old=cp.atomic_full(model,opt,out/'step-00000002.pt',meta,[{}],0)
    new=cp.atomic_full(model,opt,out/'step-00000004.pt',dict(meta,update=4),[{}],0)
    cp.retire_previous(out,new,cfg,native,'stage2',0)
    assert not old.exists()
    cp.verify_bundle(old.with_suffix('.deploy'))
    assert cp.verify_full(new)['optimizer']['state']


def test_same_schema_wrong_weights_or_identity_never_delete_full(fixture):
    out,model,opt,cfg,native,meta,x=fixture
    old=cp.atomic_full(model,opt,out/'step-00000002.pt',meta,[{}],0)
    original=cp.verify_full(old)['model_state']
    wrong={k:v.clone() for k,v in original.items()}
    wrong[next(iter(wrong))].add_(1)
    bundle=cp.export_bundle(wrong,out/'wrong.deploy',cfg,native,meta)
    with pytest.raises(ValueError,match='content'):
        cp.remove_full_after_bundle(old,bundle)
    with pytest.raises(ValueError,match='content'):
        cp.export_bundle(original,bundle,cfg,native,meta)
    other=cp.export_bundle(original,out/'other-cache.deploy',cfg,native,dict(meta,cache_identity='other'))
    with pytest.raises(ValueError,match='identity'):
        cp.remove_full_after_bundle(old,other)
    with pytest.raises(ValueError,match='identity'):
        cp.export_bundle(original,other,cfg,native,meta)
    assert old.exists()


def test_stage1_rolls_full_without_historical_optimizer(fixture):
    out,model,opt,cfg,native,meta,x=fixture
    old=cp.atomic_full(model,opt,out/'step-00000002.pt',dict(meta,stage='stage1'),[{}],0)
    new=cp.atomic_full(model,opt,out/'step-00000004.pt',dict(meta,stage='stage1',update=4),[{}],0)
    cp.retire_previous(out,new,cfg,native,'stage1',0)
    assert not old.exists() and new.exists() and not list(out.glob('*.deploy'))


def test_legacy_policy_and_unspecified_interval_are_rejected():
    with pytest.raises(ValueError):cp.checkpoint_policy({'checkpoint':{'every_epochs':1}},'stage1')
    with pytest.raises(ValueError,match='optimizer_save'):
        cp.checkpoint_policy({'checkpoint':{'format':'full-state-deploy-v2','save_every_steps':400}},'stage2')
    for interval in (None,0,-1,True,1.5):
        with pytest.raises(ValueError):cp.checkpoint_policy({'checkpoint':{'format':'full-state-deploy-v2','optimizer_save':'final_only','save_every_steps':interval}},'stage2')


def test_disk_guard_does_not_delete_previous(fixture,monkeypatch):
    out,model,opt,cfg,native,meta,x=fixture
    old=cp.atomic_full(model,opt,out/'step-00000002.pt',meta,[{}],0)
    class Empty: f_bavail=0;f_frsize=4096
    monkeypatch.setattr(cp.os,'statvfs',lambda p:Empty())
    with pytest.raises(OSError):cp.atomic_full(model,opt,out/'step-00000004.pt',dict(meta,update=4),[{}],0)
    assert old.exists()


def test_native_pi05_bundle_builds_without_original_weights(fixture):
    out,_,_,cfg,_,meta,_=fixture
    from fluxvla import register_training_components
    register_training_components()
    from fluxvla.engines import build_vla_from_cfg
    import runpy
    native=runpy.run_path(str(Path(__file__).parents[1]/'configs/pi05_subtask_fast.py'))['model']
    from fluxvla.training.subtask.config import PROMPT_CONTRACT
    cfg['prompt_contract']=PROMPT_CONTRACT
    native.update(pretrained_name_or_path=None,proj_width=16,n_action_steps=4,fast_vocab_size=8,fast_skip_tokens=2)
    for key in ('llm_backbone','llm_expert'):
        native[key].update(hidden_size=16,intermediate_size=32,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,head_dim=8,vocab_size=64)
    native['llm_expert']['adarms_cond_dim']=16
    native['vision_backbone']['vision_config'].update(hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,image_size=16,patch_size=8)
    native['projector'].update(in_dim=16,out_dim=16)
    native['action_in_proj']['out_dim']=16
    native['action_out_proj']['in_dim']=16
    for key in ('time_mlp_in','time_mlp_out'):
        native[key].update(in_dim=16,out_dim=16)
    model=build_vla_from_cfg(native).to(torch.bfloat16).eval()
    native['pretrained_name_or_path']='/this/base/does/not/exist'
    bundle=cp.export_bundle(model.state_dict(),out/'native.deploy',cfg,native,meta)
    restored=cp.build_from_bundle(bundle)
    for key,value in model.state_dict().items():
        assert torch.equal(value,restored.state_dict()[key])
        assert value.dtype==restored.state_dict()[key].dtype
    for key,value in model.named_buffers():
        assert torch.equal(value,dict(restored.named_buffers())[key])
    hidden=torch.randn(2,4,16,dtype=torch.bfloat16)
    assert torch.equal(model.project_text(hidden),restored.project_text(hidden))
    assert torch.equal(model.action_out_proj(hidden),restored.action_out_proj(hidden))


@pytest.mark.parametrize('preset',['hybrid','hybrid_insulated','action_expert'])
def test_new_prefix_contract_roundtrip(fixture,preset):
    from fluxvla.training.subtask.config import PROMPT_CONTRACT
    _,_,_,cfg,native,meta,_=fixture
    cfg['prompt_contract']=PROMPT_CONTRACT
    native['low_level_text_attention']='bidirectional'
    native['low_level_task_independent']=True
    native['rtc_training_config']=dict(enabled=True,max_delay=7,distribution='exponential',temperature=1.)
    native['loss_reduction']='local'
    meta['preset']=preset
    saved,_,contract=cp.bundle_assets(cfg,native,meta)
    assert saved['low_level_text_attention']=='bidirectional'
    assert saved['low_level_task_independent'] is True
    assert contract['prompt_contract']==PROMPT_CONTRACT
    assert contract['conditioning']['training_action_prefix']=='text_bidirectional'
    assert contract['conditioning']['high_level_text_attention']=='full_prefix_causal_outputs'
    assert contract['conditioning']['generation_is_experimental'] is False
    assert contract['low_level_action_tail']==''
    assert contract['low_level_eos'] is False
    assert contract['low_level_segmented'] is True
    assert contract['low_level_sees_task'] is False
    assert contract['all_causal'] is False
    assert contract['low_level_prompt_format']=='state_subtask'
    assert contract['training_rtc']==native['rtc_training_config']
    assert contract['loss_reduction']=='local'


def test_paper_export_rejects_model_contract_mismatch(fixture):
    from fluxvla.training.subtask.config import PROMPT_CONTRACT
    _,_,_,cfg,native,meta,_=fixture
    cfg['prompt_contract']=PROMPT_CONTRACT
    native.update(low_level_text_attention='causal',low_level_task_independent=True)
    with pytest.raises(ValueError,match='bidirectional'):cp.bundle_assets(cfg,native,meta)
    native.update(low_level_text_attention='bidirectional',low_level_task_independent=False)
    with pytest.raises(ValueError,match='task-independent'):cp.bundle_assets(cfg,native,meta)
    native['low_level_task_independent']=True
    cfg.pop('prompt_contract')
    with pytest.raises(ValueError,match='explicit paper'):cp.bundle_assets(cfg,native,meta)


def test_legacy_bundle_assets_keep_original_prompt_semantics(fixture):
    _,_,_,cfg,native,meta,_=fixture
    native['low_level_text_attention']='bidirectional'
    _,_,contract=cp.bundle_assets(cfg,native,meta)
    assert 'prompt_contract' not in contract
    assert contract['low_level_prompt_format']=='legacy'
    assert contract['low_level_sees_task'] is True
    assert contract['all_causal'] is True
    assert contract['low_level_action_tail']=='\nAction: '
