"""Unit/unit-boundary checks for the shared, hardware-free inference path."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from fluxvla.inference.subtask import SubtaskPolicy, InferenceConfig, JOINTS, GRIPPERS
from fluxvla.inference.bundle import validate_conditioning, verify_bundle_files, sha256


class Prompt:
    eos_id, pad_id = 1, 0
    def context(self, overall, bins):
        return dict(input_ids=[2,3], valid_mask=[True,True], ar_mask=[False,True])
    def decode(self, ids):
        return 'A.;\nAction: complete-token-tail'
    def low_level(self, text, bins, overall):
        self.action_text = text
        return dict(input_ids=[2,4],valid_mask=[True,True])


class Model(torch.nn.Module):
    n_action_steps, max_action_dim, attn_math = 4, 32, 'fp32'
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
    def generate_subtask(self, *args, **kwargs):
        self.generation_args = kwargs
        return dict(token_ids=torch.arange(64)[None], lengths=torch.tensor([64]), finished=[False])
    def sample_actions_given_subtask(self, *args, **kwargs):
        self.last_images = args[0].clone()
        if getattr(self, 'fail', False):
            raise RuntimeError('sampler failed')
        return torch.zeros(1, 4, 32)


def setup_policy(**kw):
    stats = dict(state=dict(q01=[-1]*14,q99=[1]*14), action=dict(q01=[0]*14,q99=[.08]*14))
    p = SubtaskPolicy(Model(), Prompt(), stats, config=InferenceConfig(**kw), image_size=16)
    state = np.arange(14,dtype=np.float32)*.01
    payload = dict(state=state,instruction='sort',images={k:np.zeros((16,16,3),np.uint8) for k in p.camera_keys})
    return p,payload


def test_absolute_actions_add_only_joints_and_preserve_full_64_tokens():
    p,payload = setup_policy()
    result = p.infer(payload)
    np.testing.assert_allclose(result['actions'][:,JOINTS],np.tile(.04+payload['state'][list(JOINTS)],(4,1)))
    np.testing.assert_allclose(result['actions'][:,GRIPPERS],.04)
    assert len(result['trace']['generated_token_ids']) == 64
    assert result['trace']['raw_generation'].endswith('complete-token-tail')
    assert p.model.generation_args['max_new_tokens'] == 64
    assert p.model.generation_args['stop_token_ids'] is None
    assert p.prompts.action_text == 'A.'


def test_hold_reuses_only_label_and_failed_action_does_not_commit():
    p,payload = setup_policy(subtask_update_interval_s=2)
    p.model.fail = True
    with pytest.raises(RuntimeError):p.infer(payload)
    assert p.stability._state is None
    p.model.fail = False
    p.infer(payload)
    payload['images'] = {k:np.full((16,16,3),255,np.uint8) for k in p.camera_keys}
    result = p.infer(payload)
    assert not result['trace']['subtask_refreshed']
    assert torch.all(p.model.last_images == 1)
    p.reset()
    assert p.stability._state is None


def test_wire_adapter_carries_deltas_and_scale_one(tmp_path):
    pytest.importorskip('soma_deploy.serving.protocol')
    from fluxvla.inference.soma_adapter import SomaPolicyAdapter
    from soma_deploy.models.pi05_subtask.client import Pi05SubtaskClientAdapter
    p,payload = setup_policy()
    p.bundle_path=str(tmp_path)
    (tmp_path/'normalization.json').write_text('{}')
    adapter=SomaPolicyAdapter(p)
    result=adapter.infer(payload)
    client=Pi05SubtaskClientAdapter(adapter.metadata)
    np.testing.assert_allclose(client.to_absolute(result),p.infer(payload)['actions'],atol=1e-8)
    assert adapter.metadata.client_config['gripper_scale'] == 1


def test_legacy_contract_requires_explicit_audited_override():
    contract=dict(state_dim=14,action_dim=14,padded_action_dim=32,low_level_prefix_source='generation')
    with pytest.raises(ValueError,match='legacy'):validate_conditioning(contract,InferenceConfig())
    validate_conditioning(contract,InferenceConfig(),allow_legacy_contract=True)
    with pytest.raises(ValueError,match='only supports'):
        validate_conditioning(contract,InferenceConfig(prefix_source='generation',subtask_switch_confirmations=1),allow_legacy_contract=True)
    contract.update(low_level_prefix_source='text',conditioning=dict(training_action_prefix='text_bidirectional'))
    validate_conditioning(contract,InferenceConfig())


def test_bundle_rejects_corruption_and_path_escape(tmp_path):
    names=['model.safetensors','model_config.json','normalization.json','inference_contract.json','tokenizer/tokenizer_config.json']
    files={}
    for name in names:
        f=tmp_path/name; f.parent.mkdir(parents=True,exist_ok=True); f.write_text('{}')
        files[name]=dict(size=f.stat().st_size,sha256=sha256(f))
    manifest=tmp_path/'DEPLOYABLE.json'
    manifest.write_text(json.dumps(dict(files=files)))
    verify_bundle_files(tmp_path)
    (tmp_path/names[0]).write_text('[]')
    with pytest.raises(ValueError,match='integrity'):verify_bundle_files(tmp_path)
    (tmp_path/names[0]).write_text('{}')
    files['../outside']={}
    manifest.write_text(json.dumps(dict(files=files)))
    with pytest.raises(ValueError,match='escapes'):verify_bundle_files(tmp_path)
