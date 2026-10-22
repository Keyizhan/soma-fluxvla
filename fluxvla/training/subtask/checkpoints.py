"""Atomic full recovery snapshots and independently loadable PI05 deployment bundles.

Only a verified deployment bundle permits removal of a historical optimizer.
No archive service or base-model weights are required by a bundle.
"""
import copy
import json
import os
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from .common import digest, write_json

GIB = 1024 ** 3


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_json(path, value):
    write_json(path, value)
    with Path(path).open('rb') as f:
        os.fsync(f.fileno())
    sync_dir(Path(path).parent)


def require_space(output, required):
    """Use available filesystem bytes, never visible-tree du or optimistic compression."""
    s = os.statvfs(output)
    available = s.f_bavail * s.f_frsize
    if available < required:
        raise OSError(f'insufficient checkpoint space: available={available}, required={required}; '
                      'no old optimizer or deployment was removed')


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def tensor_spec(state):
    return {k: dict(shape=list(v.shape), dtype=str(v.dtype).replace('torch.', ''))
            for k, v in state.items()}


def checkpoint_policy(cfg, stage):
    policy = cfg.get('checkpoint', {})
    if policy.get('format') != 'full-state-deploy-v2':
        raise ValueError('checkpoint.format must be full-state-deploy-v2; legacy epoch-full policy is disabled')
    if policy.get('optimizer_save') != 'final_only':
        raise ValueError('checkpoint.optimizer_save must be final_only; intermediate optimizer snapshots are disabled')
    interval = policy.get('save_every_steps')
    if stage == 'stage2' and (type(interval) is not int or interval <= 0):
        raise ValueError('Stage2 checkpoint.save_every_steps must be chosen as a positive optimizer-step interval')
    return policy


def verify_full(path):
    path = Path(path)
    ready = json.loads(path.with_suffix('.pt.ready.json').read_text())
    if path.stat().st_size != ready['size'] or digest(path) != ready['sha256']:
        raise ValueError(f'full checkpoint checksum failed: {path}')
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if state.get('format') != 'full-state-deploy-v2' or 'model_state' not in state:
        raise ValueError('legacy/incomplete checkpoint: full model_state is required')
    if tensor_spec(state['model_state']) != ready['model_spec']:
        raise ValueError('full checkpoint tensor schema failed')
    return state


def atomic_full(model, opt, path, metadata, rng, reserve_bytes):
    path = Path(path)
    weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    state = dict(format='full-state-deploy-v2', model_state=weights,
                 optimizer=opt.state_dict(), rng=rng, **metadata)
    # Existing full remains present while its successor and a deployment are written.
    require_space(path.parent, tensor_bytes(state) + tensor_bytes(weights) + reserve_bytes + 16*1024**2)
    partial = path.with_suffix('.pt.partial')
    torch.save(state, partial)
    with partial.open('rb') as f:
        os.fsync(f.fileno())
    # Deserialize and check keys/shapes before promoting the new full checkpoint.
    check = torch.load(partial, map_location='cpu', weights_only=True, mmap=True)
    if tensor_spec(check['model_state']) != tensor_spec(weights) or 'optimizer' not in check:
        raise ValueError('new full checkpoint failed validation; previous full was preserved')
    del check
    sha = digest(partial)
    partial.replace(path)
    sync_dir(path.parent)
    durable_json(path.with_suffix('.pt.ready.json'), dict(size=path.stat().st_size,
        sha256=sha, model_spec=tensor_spec(weights), update=metadata['update']))
    return path


def bundle_assets(cfg, model_cfg, metadata):
    from .config import PROMPT_CONTRACT, HIGH_LEVEL_ATTENTION, validate_loss_contract
    loss_contract = validate_loss_contract(cfg, model_cfg, metadata['preset'])
    if loss_contract is not None:
        recorded = metadata.get('config', cfg)
        for key in ('loss_objective', 'joint_flow_weight'):
            if recorded.get(key) != cfg[key] or metadata.get(key, cfg[key]) != cfg[key]:
                raise ValueError('checkpoint metadata loss objective differs from export configuration')
        if recorded.get('text_loss') != cfg.get('text_loss'):
            raise ValueError('checkpoint metadata text_loss differs from export configuration')
        if 'model_config' in metadata:
            validate_loss_contract(cfg, metadata['model_config'], metadata['preset'])
    paper = cfg.get('prompt_contract') == PROMPT_CONTRACT
    if cfg.get('prompt_contract') not in (None, PROMPT_CONTRACT):
        raise ValueError('unsupported deployment prompt contract')
    if not paper and model_cfg.get('low_level_task_independent') is True:
        raise ValueError('task-independent model requires explicit paper prompt contract')
    if paper and model_cfg.get('low_level_text_attention') != 'bidirectional':
        raise ValueError('paper prompt contract requires bidirectional low-level prefix')
    if paper and model_cfg.get('low_level_task_independent') is not True:
        raise ValueError('paper prompt contract requires task-independent low-level model')
    native = copy.deepcopy(model_cfg)
    native['pretrained_name_or_path'] = None
    native['low_level_prefix_source'] = 'text'
    samples = json.loads((Path(cfg['cache'])/'samples.json').read_text())
    contract = dict(version='pi05-subtask-deploy-v2', model_type='PI05Subtask',
        state_dim=14, action_dim=14, padded_action_dim=32,
        horizon=native['n_action_steps'], image_size=native['vision_backbone']['vision_config']['image_size'],
        max_prompt_length=cfg['max_prompt_length'], fast_vocab_size=native.get('fast_vocab_size'),
        fast_skip_tokens=native.get('fast_skip_tokens'),
        attn_math=native.get('attn_math'), rope_precision=native.get('rope_precision'),
        training_rtc=native.get('rtc_training_config'), loss_reduction=native.get('loss_reduction', 'global'),
        precision='preserve exact training state dtype; production bf16; CPU smoke fp32',
        task_prompt=('observation + overall_task -> subtask; observation + generated subtask -> action' if paper else
            'observation + overall_task + discretized state -> subtask; observation + overall_task + generated subtask -> action'),
        prompt_transform='PrepareSubtaskFastPrompts', state_bins=256, all_causal=not paper,
        low_level_sees_task=not paper, model_vocab_size=257152,
        prompt_variant=metadata.get('contract'), image_normalization='uint8 / 255 * 2 - 1',
        camera_aliases=cfg.get('camera_aliases'),
        low_level_prefix_source='text',
        low_level_action_tail='' if paper or native.get('low_level_text_attention')=='causal' else '\nAction: ',
        low_level_segmented=paper or native.get('low_level_text_attention')=='causal',
        low_level_prompt_format='state_subtask' if paper else 'legacy',
        low_level_eos=False,
        conditioning=dict(training_action_prefix='text_'+native.get('low_level_text_attention', 'bidirectional'),
            high_level_text_attention=HIGH_LEVEL_ATTENTION if paper else 'causal', generation_is_experimental=not paper),
        action_window=samples.get('conventions', {}).get('action_window', 'legacy-next-frame-subtask-bounded'),
        action_encoding='12 joints delta to anchor; 2 grippers absolute meters',
        normalization='train-only q01/q99; no clipping', invalid_padding=samples.get('conventions', {}).get('invalid_padding', 'masked zero'),
        state_normalization='float64_quantiles_then_float32',
        cache_identity=metadata['cache_identity'], stage=metadata['stage'], update=metadata['update'],
        preset=metadata['preset'], smoke=metadata.get('smoke', False))
    if paper:
        contract['prompt_contract'] = PROMPT_CONTRACT
    if loss_contract is not None:
        contract['training_objective'] = loss_contract
    return native, samples['normalization'], contract


def export_bundle(weights, destination, cfg, model_cfg, metadata, reserve_bytes=0):
    destination = Path(destination)
    if destination.exists():
        match_bundle(weights, destination, metadata)
        return destination
    require_space(destination.parent, tensor_bytes(weights) + reserve_bytes + 16*1024**2)
    partial = destination.with_name(destination.name+'.partial')
    # A failed export is never a valid gate; clear only this exact unpublished temp directory.
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    native, normalization, contract = bundle_assets(cfg, model_cfg, metadata)
    durable_json(partial/'model_config.json', native)
    durable_json(partial/'normalization.json', normalization)
    durable_json(partial/'inference_contract.json', contract)
    tokenizer = partial/'tokenizer'
    tokenizer.mkdir()
    names = ('tokenizer', 'special_tokens', 'added_tokens', 'vocab', 'merges', 'chat_template')
    copied = []
    for source in Path(cfg['tokenizer']).iterdir():
        if source.is_file() and source.name.startswith(names):
            shutil.copy2(source, tokenizer/source.name)
            copied.append(source.name)
    if not any(n in copied for n in ('tokenizer.json', 'tokenizer.model')):
        raise ValueError('deployment tokenizer asset missing')
    # Full state, including frozen action tensors and unused expert embedding. No delta/base dependency.
    save_file({k: v.detach().cpu().contiguous().clone() for k, v in weights.items()}, str(partial/'model.safetensors'))
    spec = tensor_spec(weights)
    with safe_open(partial/'model.safetensors', framework='pt') as f:
        actual = {k: dict(shape=list(f.get_slice(k).get_shape()),
                         dtype=str(f.get_tensor(k).dtype).replace('torch.', '')) for k in f.keys()}
    if actual != spec:
        raise ValueError('deployment tensor keys/shapes/dtypes do not match full checkpoint')
    files = {}
    for p in sorted(partial.rglob('*')):
        if p.is_file():
            with p.open('rb') as f:
                os.fsync(f.fileno())
            files[p.relative_to(partial).as_posix()] = dict(size=p.stat().st_size, sha256=digest(p))
    durable_json(partial/'DEPLOYABLE.json', dict(format='pi05-subtask-deploy-v2',
        files=files, model_spec=spec, metadata=metadata, optimizer_present=False))
    match_bundle(weights, partial, metadata)
    partial.rename(destination)
    sync_dir(destination.parent)
    return destination


def verify_bundle(path):
    path = Path(path)
    manifest = json.loads((path/'DEPLOYABLE.json').read_text())
    if manifest.get('format') != 'pi05-subtask-deploy-v2' or manifest.get('optimizer_present') is not False:
        raise ValueError('not a deployable PI05Subtask bundle')
    required = {'model.safetensors', 'model_config.json', 'normalization.json', 'inference_contract.json'}
    if not required.issubset(manifest['files']) or not any(k.startswith('tokenizer/') for k in manifest['files']):
        raise ValueError('incomplete deployment assets')
    for name, check in manifest['files'].items():
        p = path/name
        if p.stat().st_size != check['size'] or digest(p) != check['sha256']:
            raise ValueError(f'deployment checksum failed: {p}')
    with safe_open(path/'model.safetensors', framework='pt') as f:
        actual = {k: dict(shape=list(f.get_slice(k).get_shape()),
                         dtype=str(f.get_tensor(k).dtype).replace('torch.', '')) for k in f.keys()}
    if actual != manifest['model_spec']:
        raise ValueError('deployment tensor schema failed')
    return manifest


def match_bundle(weights, path, metadata):
    manifest = verify_bundle(path)
    keys = ('stage', 'update', 'cache_identity', 'config', 'model_config', 'preset', 'smoke', 'total_updates')
    if any(manifest['metadata'].get(k) != metadata.get(k) for k in keys):
        raise ValueError('deployment identity/config does not match source checkpoint')
    if manifest['model_spec'] != tensor_spec(weights):
        raise ValueError('deployment tensor schema does not match source checkpoint')
    with safe_open(Path(path)/'model.safetensors', framework='pt') as f:
        for key, expected in weights.items():
            if not torch.equal(expected.detach().cpu(), f.get_tensor(key)):
                raise ValueError(f'deployment tensor content does not match source checkpoint: {key}')
    return manifest


def load_bundle(model, path):
    manifest = verify_bundle(path)
    model.load_state_dict(load_file(str(Path(path)/'model.safetensors')), strict=True)
    return manifest


def build_from_bundle(path, device='cpu'):
    """Build a native PI05Subtask directly from the bundle, without original weights."""
    verify_bundle(path)
    from fluxvla import register_training_components
    register_training_components()
    from fluxvla.engines import build_vla_from_cfg
    native = json.loads((Path(path)/'model_config.json').read_text())
    if native.get('pretrained_name_or_path') is not None:
        raise ValueError('deployment must not depend on original base weights')
    # Keep computed, nonpersistent buffers (position IDs / RoPE frequencies) real.
    # Putting the entire constructor on meta would omit these from state_dict.
    from accelerate import init_empty_weights
    with init_empty_weights(include_buffers=False):
        model = build_vla_from_cfg(native)
    model.load_state_dict(load_file(str(Path(path)/'model.safetensors')), strict=True, assign=True)
    dtypes = {p.dtype for p in model.parameters() if p.is_floating_point()}
    # The trainer casts the whole model, including nonpersistent floating buffers.
    if len(dtypes) == 1:
        model = model.to(device=device, dtype=next(iter(dtypes)))
    else:
        model = model.to(device=device)
    return model.eval()


def remove_full_after_bundle(path, bundle):
    """The deployment checksum+schema gate replaces the old external archive gate."""
    state = verify_full(path)
    match_bundle(state['model_state'], bundle, state)
    del state
    Path(path).unlink()
    Path(path).with_suffix('.pt.ready.json').unlink(missing_ok=True)
    sync_dir(Path(path).parent)


def retire_previous(out, latest, cfg, model_cfg, stage, reserve_bytes):
    for old in sorted(Path(out).glob('step-*.pt')):
        if old == latest:
            continue
        if stage == 'stage1':
            # A verified new full state is the rolling Stage1 recovery gate.
            verify_full(latest)
            old.unlink()
            old.with_suffix('.pt.ready.json').unlink(missing_ok=True)
        else:
            state = verify_full(old)
            bundle = export_bundle(state['model_state'], old.with_suffix('.deploy'), cfg,
                                   model_cfg, {k:v for k,v in state.items()
                                   if k not in ('model_state','optimizer','rng','format')}, reserve_bytes)
            del state
            remove_full_after_bundle(old, bundle)
    sync_dir(out)
