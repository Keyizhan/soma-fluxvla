from pathlib import Path
import pytest
import torch
from fluxvla.training.subtask.config import load_config,check_comparison
from fluxvla.training.subtask.randomness import flow_random

ROOT=Path(__file__).resolve().parents[1]
def configs():
    return [load_config(ROOT/'configs/subtask'/name) for name in ['fabric106.json','fabric106_hybrid_insulated.json','fabric106_two_stage.json']]

def test_resolved_methods_only_allowlisted_differences():
    cfg=configs();check_comparison(cfg)
    cfg[2]['stages'][-1]['peak_lr']*=2
    with pytest.raises(ValueError,match='non-allowlisted'):check_comparison(cfg)

def test_stateless_flow_randomness():
    actions=torch.zeros(2,4,32);initial=torch.get_rng_state()
    n,t=flow_random(['a','b'],actions,1702,'train',1)
    assert torch.equal(initial,torch.get_rng_state())
    n2,t2=flow_random(['b','a'],actions,1702,'train',1)
    assert torch.equal(n[0],n2[1]) and torch.equal(t[0],t2[1])
    flow_random(['a','b'],actions,1702,'validation')
    n3,t3=flow_random(['a','b'],actions,1702,'train',1)
    assert torch.equal(n,n3) and torch.equal(t,t3)
    assert not torch.equal(n,flow_random(['a','b'],actions,1702,'train',2)[0])

def test_zero_weight_metrics_are_real():
    from test_subtask_fast_model import tiny_config,layout,image_batch
    from fluxvla.engines import build_vla_from_cfg
    from fluxvla.training.subtask.train import Loss
    from fluxvla.training.subtask.fixtures import register_training_components
    register_training_components()
    model=build_vla_from_cfg(tiny_config(hl_loss_weight=0.,fast_loss_weight=0.,action_loss_weight=1.)).eval()
    high=layout();images,masks=image_batch()
    batch={k.replace('hl_','high_'):v for k,v in high.items()}
    batch.update(images=images,img_masks=masks,real=torch.ones(2,dtype=torch.bool),
        low_input_ids=torch.tensor([[2,3,4],[2,3,4]]),low_valid_mask=torch.ones(2,3,dtype=torch.bool),
        actions=torch.zeros(2,4,32),action_masks=torch.ones(2,4,32,dtype=torch.bool))
    with torch.no_grad():_,metrics=Loss(model,True,metrics=True)(batch,torch.ones(3))
    assert metrics.shape==(3,) and torch.isfinite(metrics).all() and (metrics>0).all()


def test_no_augmentation_or_ema(tmp_path):
    import json
    for field in ('augmentation','ema'):
        path=tmp_path/'bad.json';path.write_text(json.dumps({field:True}))
        with pytest.raises(ValueError):load_config(path)

def test_fixed_split_rejects_duplicate_trajectory(tmp_path):
    import json
    import shutil
    from test_subtask_pipeline import synthetic
    from fluxvla.training.subtask.prepare import scan
    cfg=synthetic(tmp_path)
    source=Path(cfg['incoming'])/'lerobot_v2_1/data'
    shutil.copyfile(source/'episode_000000.parquet',source/'episode_000001.parquet')
    with pytest.raises(ValueError,match='duplicate trajectory'):scan(cfg)

def test_holdout_never_enters_cache_or_prompt_budget(tmp_path):
    import json
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from test_subtask_pipeline import synthetic,CODE
    from fluxvla.training.subtask.prepare import prepare as prepare_cache
    cfg=synthetic(tmp_path);root=Path(cfg['incoming'])/'lerobot_v2_1'
    table=pq.read_table(root/'data/episode_000001.parquet').to_pydict()
    table['state']=(np.asarray(table['state'])+100).tolist()
    table['subtask']=['holdout '*1000]*12
    pq.write_table(pa.table(table),root/'data/episode_000002.parquet')
    info=json.loads((root/'meta/info.json').read_text());info.update(total_episodes=3,total_frames=36)
    (root/'meta/info.json').write_text(json.dumps(info))
    with (root/'meta/episodes.jsonl').open('a') as f:f.write(json.dumps(dict(episode_index=2,length=12))+'\n')
    Path(cfg['split_manifest']).write_text(json.dumps(dict(episodes={'0':'train','1':'validation','2':'test'})))
    cfg['expected_episodes']=3
    cache=prepare_cache(cfg,CODE)
    samples=json.loads((cache/'samples.json').read_text())
    assert len(samples['records'])==24
    assert {x['split'] for x in samples['records']}=={'train','validation'}
    assert not any('holdout' in x['subtask'] for x in samples['records'])
    receipt=json.loads((cache/'READY.json').read_text())
    assert len(receipt['data_identity'])==3
    cfg['seed']+=1
    from fluxvla.training.subtask.prepare import verify
    verify(cfg)  # fixed data cache is reusable across random seeds


def test_extends_method_resolves_seed_only_after_merging(tmp_path):
    import json
    path=tmp_path/'run.json'
    path.write_text(json.dumps(dict(extends=str(ROOT/'configs/subtask/fabric106.json'),seed=99,training_method='two_stage')))
    cfg=load_config(path)
    assert cfg['output'].endswith('seed-99')
    assert [s['name'] for s in cfg['stages']]==['stage1','stage2']
    assert cfg['stages'][-1]['preset']=='action_expert'

def test_methods_have_intended_initialization_budgets():
    hybrid,insulated,two=configs()
    assert [s['epochs'] for s in hybrid['stages']]==[8]
    assert [s['epochs'] for s in insulated['stages']]==[4,8]
    assert [s['epochs'] for s in two['stages']]==[4,8]


def test_staged_methods_share_stage1_identity():
    from fluxvla.training.subtask.config import stage1_identity
    _,insulated,two=configs()
    assert stage1_identity(insulated)==stage1_identity(two)
    two['stages'][0]['peak_lr']*=2
    with pytest.raises(ValueError,match='shared stage1'):
        check_comparison([insulated,two])


def test_paper_contract_is_shared_and_old_training_contract_rejected():
    from fluxvla.training.subtask.config import PROMPT_CONTRACT,CACHE_VERSION,resolve_config
    for cfg in configs():
        assert cfg['prompt_contract']==PROMPT_CONTRACT
        assert cfg['version']==CACHE_VERSION
        assert cfg['output']==f"work_dirs/goal_text_v1_paper_prefix_token_ce_v1_{cfg['training_method']}/seed-{cfg['seed']}"
        assert cfg['cache'].endswith('goal_text_v1_paper_prefix')
        assert cfg['shared_stage1_root'].endswith('shared_goal_text_v1_paper_prefix_token_ce_v1')
    with pytest.raises(ValueError,match='prompt_contract'):
        resolve_config(dict(prompt_contract='legacy'))


def test_paper_cache_rejects_missing_or_changed_prompt_identity():
    from fluxvla.training.subtask.config import PROMPT_CONTRACT,CACHE_VERSION
    from fluxvla.training.subtask.prepare import validate_cache_contract,WINDOW_CONTRACT,SAMPLING_CONTRACT
    ready=dict(version=CACHE_VERSION,prompt_contract=PROMPT_CONTRACT,
               window_contract=WINDOW_CONTRACT,sampling=SAMPLING_CONTRACT)
    validate_cache_contract(ready)
    for bad in (dict(ready,prompt_contract='legacy'),dict(ready,version='controlled-goal-text-v2'),
                {k:v for k,v in ready.items() if k!='prompt_contract'}):
        with pytest.raises(ValueError,match='paper prompt'):validate_cache_contract(bad)
