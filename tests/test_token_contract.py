"""Unified objective identity, configurable lambda and compatible data caches."""
import copy
import json
from pathlib import Path

import pytest
import torch

from fluxvla.training.subtask.config import (
    CACHE_VERSION, LOSS_OBJECTIVE, PROMPT_CONTRACT, check_comparison,
    load_config, resolve_config, stage1_identity, validate_loss_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def configurations():
    return [load_config(p) for p in sorted((ROOT / 'configs/subtask').glob('fabric106*.json'))]


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True, None, '1'])
def test_invalid_lambda_rejected(value):
    with pytest.raises(ValueError, match='joint_flow_weight'):
        resolve_config(dict(joint_flow_weight=value))


def test_objective_is_explicit_and_lambda_shared():
    configs = configurations()
    check_comparison(configs)
    for cfg in configs:
        assert cfg['loss_objective'] == LOSS_OBJECTIVE
        assert cfg['joint_flow_weight'] == 1.
        assert 'token_ce_v1' in cfg['shared_stage1_root']
        assert 'token_ce_v1' in cfg['output']
        assert cfg['version'] == CACHE_VERSION
        assert cfg['cache'] == 'cache/goal_text_v1_paper_prefix'
    configs[0]['joint_flow_weight'] = .25
    with pytest.raises(ValueError, match='non-allowlisted'):
        check_comparison(configs)
    with pytest.raises(ValueError, match='loss_objective'):
        resolve_config(dict(loss_objective='separate-ce'))


def test_old_stage1_objective_cannot_match_new_identity():
    cfg = next(c for c in configurations() if c['training_method'] == 'two_stage')
    old = copy.deepcopy(cfg)
    old.pop('loss_objective')
    old.pop('joint_flow_weight')
    # Even when paths are manually equalized, the receipt's old training
    # identity cannot match the new objective and pass the existing resume gate.
    assert stage1_identity(old) != stage1_identity(cfg)


def test_objective_change_reuses_identical_data_cache():
    from fluxvla.training.subtask.common import digest
    from fluxvla.training.subtask.prepare import (
        SAMPLING_CONTRACT, WINDOW_CONTRACT, validate_cache_config,
    )
    cfg = configurations()[0]
    old = copy.deepcopy(cfg)
    old.pop('loss_objective')
    old.pop('joint_flow_weight')
    old['shared_stage1_root'] = 'work_dirs/shared_goal_text_v1_paper_prefix'
    ready = dict(config=old, version=CACHE_VERSION, prompt_contract=PROMPT_CONTRACT,
                 window_contract=WINDOW_CONTRACT, sampling=SAMPLING_CONTRACT,
                 split_manifest_sha256=digest(cfg['split_manifest']))
    validate_cache_config(ready, cfg)


@pytest.mark.parametrize('preset,token,flow', [
    ('stage1_fast', 1., 0.), ('hybrid', 1., .25),
    ('hybrid_insulated', 1., .25), ('action_expert', 0., 1.),
])
def test_model_factory_applies_lambda_only_to_joint(preset, token, flow, monkeypatch):
    from fluxvla.training.subtask import fixtures

    class Dummy:
        def to(self, **kwargs):
            return self

    monkeypatch.setattr(fixtures, 'build_vla_from_cfg', lambda cfg: Dummy())
    _, native = fixtures.build_model(preset, True, torch.device('cpu'), joint_flow_weight=.25)
    assert native['text_loss_mode'] == 'unified_token'
    assert native['token_loss_weight'] == token
    assert native['hl_loss_weight'] == native['fast_loss_weight'] == 0.
    assert native['action_loss_weight'] == flow
    cfg = dict(loss_objective=LOSS_OBJECTIVE, joint_flow_weight=.25)
    contract = validate_loss_contract(cfg, native, preset)
    assert contract['flow_loss_weight'] == flow


def test_export_records_and_checks_objective(tmp_path):
    from fluxvla.training.subtask.checkpoints import bundle_assets
    cfg = configurations()[0]
    cfg.update(cache=str(tmp_path), joint_flow_weight=.25)
    (tmp_path / 'samples.json').write_text(json.dumps(dict(normalization={})))
    native = dict(type='PI05Subtask', n_action_steps=50,
                  vision_backbone=dict(vision_config=dict(image_size=224)),
                  low_level_text_attention='bidirectional', low_level_task_independent=True,
                  text_loss_mode='unified_token', token_loss_weight=1.,
                  hl_loss_weight=0., fast_loss_weight=0., action_loss_weight=.25)
    metadata = dict(preset='hybrid', cache_identity='fixture', stage='stage2', update=1)
    _, _, contract = bundle_assets(cfg, native, metadata)
    assert contract['training_objective']['name'] == LOSS_OBJECTIVE
    assert contract['training_objective']['flow_loss_weight'] == .25
    for change in (dict(text_loss_mode='separate'), dict(hl_loss_weight=.15),
                   dict(token_loss_weight=0.), dict(action_loss_weight=1.)):
        with pytest.raises(ValueError, match='loss settings'):
            bundle_assets(cfg, dict(native, **change), metadata)
    for bad_metadata in (dict(metadata, joint_flow_weight=1.),
                         dict(metadata, config=dict(cfg, loss_objective='separate-ce')),
                         dict(metadata, model_config=dict(native, action_loss_weight=1.))):
        with pytest.raises(ValueError, match='loss'):
            bundle_assets(cfg, native, bad_metadata)
    legacy_cfg = dict(cfg)
    legacy_cfg.pop('loss_objective')
    with pytest.raises(ValueError, match='explicit loss_objective'):
        bundle_assets(legacy_cfg, native, metadata)
    legacy_model = dict(native, text_loss_mode='separate', hl_loss_weight=.15,
                        fast_loss_weight=.15, token_loss_weight=0., action_loss_weight=1.)
    _, _, old_contract = bundle_assets(legacy_cfg, legacy_model, metadata)
    assert 'training_objective' not in old_contract
