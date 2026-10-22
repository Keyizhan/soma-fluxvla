"""Loss gate configuration, export identity and train-only mask integration."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fluxvla.training.subtask.config import load_config, resolve_config, canonical_prompts
from fluxvla.training.subtask.checkpoints import bundle_assets
from fluxvla.training.subtask.fixtures import prepare
from fluxvla.training.subtask.text_loss import model_loss_settings, weights_at
from test_template_masks import OffsetTokenizer

ROOT = Path(__file__).resolve().parents[1]


def test_example_configs_isolate_outputs_and_have_intended_weights():
    baseline = load_config(ROOT/'configs/subtask/pro6000_hybrid_insulated_la10.json')
    configs = {p.stem: load_config(p) for p in (ROOT/'configs/subtask/loss_examples').glob('*.json')}
    assert len(configs) == 5
    assert len({c['output'] for c in configs.values()}) == 5
    assert len({c['shared_stage1_root'] for c in configs.values()}) == 5
    for c in configs.values():
        assert c['output'] != baseline['output']
        assert c['shared_stage1_root'] != baseline['shared_stage1_root']
        assert c['cache'] == baseline['cache']
        assert c['joint_flow_weight'] == baseline['joint_flow_weight']
    c = configs['separate_split_linear']['text_loss']
    assert weights_at(c, 1, 101) == (.5, .5)
    assert weights_at(c, 101, 101) == pytest.approx((1/11, 10/11))


def test_fit_uses_training_labels_only_and_never_validation_labels():
    labels = ['Move red cloth', 'Move blue cloth', 'Unseen evaluation sentence']
    class Fixture(SimpleNamespace):
        def __len__(self):
            return 3
    fx = Fixture(states=np.zeros((3, 14)), fast_ids=[[1], [2], [3]],
                         img_masks=np.ones((3, 3), dtype=bool),
                         texts=lambda i: ('Task', labels[i]),
                         subset=lambda s: [0, 1] if s == 'train' else [2])
    prompts = canonical_prompts(OffsetTokenizer(), dict(max_prompt_length=512))
    packed = prepare(fx, prompts, 14)
    from fluxvla.transforms.template_masks import normalize
    assert normalize(labels[2]) not in prompts.subtask_mask_spec.records
    p = packed[2]['high']
    assert np.array_equal(p['subtask_key_mask'], p['subtask_region_mask'])
    assert not p['subtask_template_mask'].any()
    assert packed[0]['high']['subtask_template_mask'].any()


def test_export_keeps_schedule_and_rejects_changed_metadata(tmp_path):
    c = resolve_config(dict(text_loss=dict(mode='separate', weighting='linear',
        split_subtask=True), joint_flow_weight=10.))
    c.update(cache=str(tmp_path), max_prompt_length=256)
    (tmp_path/'samples.json').write_text(json.dumps(dict(normalization={})))
    native = dict(type='PI05Subtask', n_action_steps=50,
        vision_backbone=dict(vision_config=dict(image_size=224)),
        low_level_text_attention='bidirectional', low_level_task_independent=True,
        action_loss_weight=10., **model_loss_settings(c['text_loss'], 'hybrid_insulated'))
    metadata = dict(preset='hybrid_insulated', cache_identity='fixture', stage='stage2',
                    update=51, config=copy.deepcopy(c), model_config=native)
    _, _, contract = bundle_assets(c, native, metadata)
    assert contract['training_objective']['text_loss'] == c['text_loss']
    assert contract['training_objective']['weight_normalization'] == 'none'
    metadata['config']['text_loss']['end_fast_weight'] = 20.
    with pytest.raises(ValueError, match='text_loss'):
        bundle_assets(c, native, metadata)
