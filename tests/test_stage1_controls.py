import json
from pathlib import Path
from types import SimpleNamespace

from fluxvla.training.subtask import run


def config(tmp_path):
    cache=tmp_path/'cache';cache.mkdir()
    (cache/'samples.json').write_text(json.dumps({'records':[{'split':'train'}]*128}))
    return dict(cache=str(cache),output=str(tmp_path/'out'),shared_stage1_root=str(tmp_path/'shared'),seed=1,
        global_batch=64,expected_gpus=2,training_method='hybrid_insulated',
        checkpoint=dict(save_every_steps=2,safety_reserve_gib=20),
        stages=[dict(name='stage1',preset='stage1_fast',epochs=3,eval_at_epochs=[1,2,3],keep_epoch_checkpoints=True),
                dict(name='stage2',preset='hybrid_insulated',epochs=2)])


def test_storage_stage1_only_retains_three_epochs_plus_final(tmp_path,monkeypatch):
    cfg=config(tmp_path)
    monkeypatch.setattr(run.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    plan=run.checkpoint_storage_plan(cfg,'stage1')
    assert plan['total_optimizer_updates']==6
    assert plan['stage2_save_steps']==[]
    assert plan['stage2_total_updates']==0
    assert plan['stage1_retained_epoch_bundles']==3
    assert plan['peak_reserved_bytes']==48*1024**3
    full=run.checkpoint_storage_plan(cfg)
    assert full['total_optimizer_updates']==10
    assert full['peak_reserved_bytes']==83*1024**3


def test_retention_keeps_every_candidate_and_legacy_cleanup_reports_all(tmp_path):
    from fluxvla.training.subtask.train import cleanup_stage1_candidates
    candidates=[]
    for epoch in range(1,4):
        p=tmp_path/f'epoch-{epoch}';p.mkdir();(p/'weight').write_text('test')
        candidates.append((.1,epoch,epoch*2,p))
    assert cleanup_stage1_candidates(candidates,True)==[]
    assert all(p.exists() for _,_,_,p in candidates)
    assert cleanup_stage1_candidates(candidates)==[1,2,3]
    assert not any(p.exists() for _,_,_,p in candidates)


def test_stop_after_stage1_writes_receipt_without_launching_stage2(tmp_path,monkeypatch):
    from fluxvla.training.subtask import common,config as config_module,checkpoints
    cfg=config(tmp_path)
    shared=Path(cfg['shared_stage1_root'])/'seed-1';stage1=shared/'stage1';stage1.mkdir(parents=True)
    deployment=stage1/'final.deploy';deployment.mkdir()
    (stage1/'COMPLETE.json').write_text(json.dumps(dict(deployment=str(deployment))))
    monkeypatch.setattr(run,'load_config',lambda path:cfg)
    monkeypatch.setattr(run,'preflight',lambda *a,**k:None)
    monkeypatch.setattr(run,'checkpoint_storage_plan',lambda *a:None)
    monkeypatch.setattr(common,'validate_training_method',lambda cfg:None)
    monkeypatch.setattr(config_module,'stage1_identity',lambda cfg:'identity')
    monkeypatch.setattr(run,'digest',lambda path:'digest')
    monkeypatch.setattr(checkpoints,'verify_bundle',lambda p:dict(metadata=dict(cache_identity='digest',stage1_identity='identity')))
    commands=[]
    monkeypatch.setattr(run.subprocess,'run',lambda command,**kwargs:commands.append(command))
    monkeypatch.setattr(run.sys,'argv',['run','--config','ignored.json','--stop-after-stage','stage1'])
    run.main()
    out=Path(cfg['output'])
    assert (out/'stage1-initialization.json').exists()
    assert (shared/'selected-stage1.json').exists()
    assert json.loads((out/'STOPPED_AFTER_STAGE.json').read_text())['stage']=='stage1'
    assert not (out/'COMPLETE.json').exists()
    assert not any('fluxvla.training.subtask.train' in c for c in commands)
