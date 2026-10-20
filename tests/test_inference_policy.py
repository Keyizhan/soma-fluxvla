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
    from soma_deploy.serving.ws_server import PolicyServer
    from soma_deploy.serving import msgpack_numpy
    from websockets.sync.client import connect
    server=PolicyServer(adapter.metadata,adapter.infer).start_in_thread()
    try:
        with connect(f'ws://127.0.0.1:{server.bound_port}',open_timeout=5) as ws:
            metadata=msgpack_numpy.unpackb(ws.recv(timeout=5))
            assert metadata['client_adapter']=='pi05_subtask'
            ws.send(msgpack_numpy.packb(dict(type='infer',request_id='offline-test',payload=payload)))
            reply=msgpack_numpy.unpackb(ws.recv(timeout=5))
            assert reply['ok'], reply
            np.testing.assert_allclose(client.to_absolute(reply['data']),p.infer(payload)['actions'],atol=1e-8)
    finally:
        server.stop()


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


def test_new_export_state_arithmetic_matches_training_without_clipping():
    from fluxvla.training.subtask.common import normalize
    from fluxvla.inference.preprocessing import normalize_state
    rng=np.random.default_rng(1702)
    stats=dict(q01=rng.normal(size=14).tolist(),q99=(rng.normal(size=14)+10).tolist())
    states=rng.normal(size=(10,14)).astype(np.float32)*10
    for state in states:
        np.testing.assert_array_equal(normalize_state(state,stats,quantile_dtype=np.float64),normalize(state,stats))


def test_offline_evaluation_reports_physical_errors_without_using_gt_labels(tmp_path):
    import runpy
    from pathlib import Path
    evaluate=runpy.run_path(str(Path(__file__).parents[1]/'scripts/evaluate_subtask.py'))['evaluate']
    p,payload=setup_policy(subtask_switch_confirmations=1)
    target=p.infer(payload)['actions'].copy()
    target[:,JOINTS] += .1
    target[:,GRIPPERS] += .02
    target[-1] = 99
    mask=np.ones_like(target,dtype=bool);mask[-1]=False
    arrays=dict(states=np.stack([payload['state']]*2),images=np.zeros((2,9,16,16),np.uint8),
                actions=np.stack([target]*2),action_masks=np.stack([mask]*2))
    records=[dict(episode=0,frame=i,overall_task='sort',subtask='GT label not generated') for i in range(2)]
    result=evaluate(p,arrays,records,tmp_path/'evaluation')
    assert result['exact_subtask_accuracy']==0
    assert result['joint_rmse_rad']==pytest.approx(.1,abs=1e-7)
    assert result['gripper_mae_m']==pytest.approx(.02,abs=1e-7)
    assert result['within_gt_interval_comparisons']==1
    events=[json.loads(line) for line in (tmp_path/'evaluation/predictions.jsonl').read_text().splitlines()]
    assert all(e['subtask']=='A.;' and len(e['trace']['generated_token_ids'])==64 for e in events)
