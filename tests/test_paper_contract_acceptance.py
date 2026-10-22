"""Independent integration acceptance for paper conditioning and prefix attention.

Uses real tokenizer assets, model weights, bundle serialization and policy loading.
On Windows only the POSIX filesystem durability hooks are replaced; this suite does
not claim to validate production filesystem crash durability.
"""
import copy
import json
import os
import shutil

import numpy as np
import pytest
import torch

from fluxvla.engines import build_tokenizer_from_cfg, build_vla_from_cfg
from fluxvla.inference.subtask import InferenceConfig, SubtaskPolicy
from fluxvla.training.presets.pi05_subtask_fast import PRESETS
from fluxvla.training.subtask import checkpoints as cp
from fluxvla.training.subtask.config import canonical_prompts, load_config
from test_subtask_fast_model import tiny_config


def real_tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ['<pad>', '<eos>', '<bos>', '<unk>', 'Task', 'State', 'Subtask',
             ':', ';', '.', 'fold', 'sort', 'grasp', 'place', 'cloth', '128']
    tokenizer = Tokenizer(models.WordLevel(dict(zip(words, range(len(words)))),
                                           unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='<pad>',
                           eos_token='<eos>', bos_token='<bos>',
                           unk_token='<unk>').save_pretrained(path)
    return build_tokenizer_from_cfg(dict(type='PretrainedTokenizer', model_path=str(path)))


def tiny_recipe(preset):
    architecture = {'llm_backbone', 'llm_expert', 'vision_backbone', 'projector',
                    'action_in_proj', 'action_out_proj', 'time_mlp_in', 'time_mlp_out',
                    'proj_width', 'n_action_steps', 'pretrained_name_or_path'}
    overrides = {key: copy.deepcopy(value) for key, value in PRESETS[preset]['model'].items()
                 if key not in architecture}
    overrides.update(fast_vocab_size=8, fast_skip_tokens=4, attn_math='fp32')
    return tiny_config(**overrides)


@pytest.mark.parametrize('method,preset', [
    ('hybrid', 'hybrid'),
    ('hybrid_insulated', 'hybrid_insulated'),
    ('two_stage', 'action_expert'),
])
def test_real_bundle_policy_preserves_task_independence(tmp_path, monkeypatch,
                                                       method, preset):
    from pathlib import Path

    config_dir = Path(__file__).resolve().parents[1] / 'configs/subtask'
    # Select by resolved method so names remain independent of compatibility filenames.
    configs = [load_config(p) for p in config_dir.glob('fabric106*.json')]
    cfg = next(c for c in configs if c['training_method'] == method)
    cache = tmp_path / 'cache'
    cache.mkdir()
    statistics = {name: dict(q01=[-1.] * 14, q99=[1.] * 14)
                  for name in ('state', 'action')}
    (cache / 'samples.json').write_text(json.dumps(dict(normalization=statistics)))
    tokenizer_dir = tmp_path / 'tokenizer'
    tokenizer = real_tokenizer(tokenizer_dir)
    cfg.update(cache=str(cache), tokenizer=str(tokenizer_dir))
    prompts = canonical_prompts(tokenizer, cfg)
    states = np.zeros(14, dtype=np.float32)
    low_a = prompts.low_level('grasp cloth', states, 'fold')
    low_b = prompts.low_level('grasp cloth', states, 'sort')
    np.testing.assert_array_equal(low_a['input_ids'], low_b['input_ids'])
    assert not np.array_equal(prompts.context('fold', states)['input_ids'],
                              prompts.context('sort', states)['input_ids'])
    assert not prompts.context('fold', states)['ar_mask'].any()

    torch.manual_seed(714)
    native = tiny_recipe(preset)
    assert native['low_level_text_attention'] == 'bidirectional'
    model = build_vla_from_cfg(native).float().eval()
    metadata = dict(stage='stage2', preset=preset, update=1, total_updates=1,
                    config=cfg, model_config=native, cache_identity='acceptance',
                    contract=prompts.variant, smoke=True)
    if os.name == 'nt':
        from types import SimpleNamespace

        monkeypatch.setattr(cp, 'sync_dir', lambda path: None)
        # Windows rejects fsync on the read-only handles used by the POSIX
        # production writer. Replace cp's reference, never the global os module.
        monkeypatch.setattr(cp, 'os', SimpleNamespace(fsync=lambda fd: None))

        def available_space(path, required):
            assert shutil.disk_usage(path).free >= required

        monkeypatch.setattr(cp, 'require_space', available_space)
    bundle = cp.export_bundle(model.state_dict(), tmp_path / 'final.deploy',
                              cfg, native, metadata)
    cp.verify_bundle(bundle)
    policy = SubtaskPolicy.from_bundle(bundle, device='cpu',
                                       config=InferenceConfig(num_denoise_steps=2))
    assert policy.prompts.low_level_sees_task is False
    assert policy.prompts.all_causal is False
    assert policy.model.low_level_text_attention == 'bidirectional'
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, policy.model.state_dict()[key], rtol=0, atol=0)
    deployed_low = policy.prompts.low_level('grasp cloth', states, 'sort')
    np.testing.assert_array_equal(low_a['input_ids'], deployed_low['input_ids'])
    np.testing.assert_array_equal(low_a['valid_mask'], deployed_low['valid_mask'])

    # Hold the generated subtask fixed, not the overall-task input. Real action
    # inference must then be invariant, including its image/prefix cache path.
    generated = tokenizer.encode('grasp cloth;', add_special_tokens=False)
    captured = []
    action_prefixes = []
    original_sampler = policy.model.sample_actions_given_subtask

    def capture_action_prefix(*args, **kwargs):
        action_prefixes.append(args[2].detach().cpu().numpy()[0].copy())
        assert kwargs.get('prefix_cache') is None
        return original_sampler(*args, **kwargs)

    def fixed_generation(images, ids, valid, img_masks, **kwargs):
        captured.append(ids.detach().clone())
        assert kwargs.get('return_cache', False) is False
        return dict(token_ids=torch.tensor([generated]), lengths=torch.tensor([len(generated)]),
                    finished=torch.tensor([True]))

    monkeypatch.setattr(policy.model, 'generate_subtask', fixed_generation)
    monkeypatch.setattr(policy.model, 'sample_actions_given_subtask', capture_action_prefix)
    payload = dict(state=states, instruction='fold', images={
        name: np.full((16, 16, 3), 90, dtype=np.uint8) for name in policy.camera_keys})
    noise = torch.randn(1, 4, 32)
    result_a = policy.infer(payload, noise=noise)
    policy.reset()
    result_b = policy.infer(dict(payload, instruction='sort'), noise=noise)
    assert not torch.equal(captured[0], captured[1])
    np.testing.assert_array_equal(low_a['input_ids'], action_prefixes[0])
    np.testing.assert_array_equal(low_a['input_ids'], action_prefixes[1])
    np.testing.assert_array_equal(result_a['normalized_actions'], result_b['normalized_actions'])
    assert result_a['subtask'] == result_b['subtask']
    # Invariance must not pass because the policy ignores all language.
    generated[:] = tokenizer.encode('place cloth;', add_special_tokens=False)
    policy.reset()
    result_c = policy.infer(payload, noise=noise)
    assert not np.array_equal(result_a['normalized_actions'], result_c['normalized_actions'])
    with pytest.raises(ValueError):
        SubtaskPolicy.from_bundle(bundle, device='cpu', config=InferenceConfig(
            prefix_source='generation', subtask_switch_confirmations=1))
