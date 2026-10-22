"""Resolved configuration and strict controlled-comparison contracts."""
import copy
import json
from pathlib import Path

PROMPT_CONTRACT = 'pi05-subtask-paper-v1'
CACHE_VERSION = 'controlled-goal-text-paper-v3'
HIGH_LEVEL_ATTENTION = 'full_prefix_causal_outputs'

def merge(base, override):
    result=copy.deepcopy(base)
    for key,value in override.items():
        result[key]=merge(result[key],value) if isinstance(value,dict) and isinstance(result.get(key),dict) else copy.deepcopy(value)
    return result

def _load_raw(path):
    path=Path(path)
    raw=json.loads(path.read_text())
    parent=raw.pop('extends',None)
    return merge(_load_raw(path.parent/parent),raw) if parent else raw

def load_config(path):
    return resolve_config(_load_raw(path))

def resolve_config(result):
    result=copy.deepcopy(result)
    result.setdefault('prompt_contract', PROMPT_CONTRACT)
    if result['prompt_contract'] != PROMPT_CONTRACT:
        raise ValueError('unsupported training prompt_contract; use the paper conditioning contract')
    if 'version' in result and result['version'] != CACHE_VERSION:
        raise ValueError('training cache version predates paper prompt contract; use a new cache')
    # MUST use true Beta for ordinary/subtask parity; never accept legacy fallback.
    result.setdefault('time_sampler','beta')
    for key, default in (('time_beta_alpha', 1.5), ('time_beta_beta', 1.0)):
        value = result.setdefault(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float('inf'):
            raise ValueError(f'{key} must be finite and positive')
    if result['time_sampler'] != 'beta':raise ValueError("time_sampler must be 'beta'")
    if result.get('augmentation',False) or result.get('ema',False):raise ValueError('controlled comparison requires no augmentation and no EMA')
    if result.get('dtype','bf16')!='bf16':raise ValueError('controlled comparison requires bf16')
    if 'training_method' in result:
        presets={'hybrid':'hybrid','hybrid_insulated':'hybrid_insulated','two_stage':'action_expert'}
        method=result['training_method']
        if method not in presets:raise ValueError('unknown training_method')
        next(stage for stage in result['stages'] if stage['name']=='stage2')['preset']=presets[method]
        if method=='hybrid':result['stages']=[stage for stage in result['stages'] if stage['name']!='stage1']
    if 'output' in result:result['output']=result['output'].format(seed=result['seed'],method=result.get('training_method',''))
    return result

def common_identity(cfg):
    result=copy.deepcopy(cfg)
    for key in ('output','notes','training_method','run_id'):
        result.pop(key,None)
    result['stages']=[stage for stage in result['stages'] if stage['name']!='stage1']
    for stage in result['stages']:
        if stage['name']=='stage2': stage.pop('preset',None)
    return result

def check_comparison(configs):
    if len({c['training_method'] for c in configs})!=len(configs):
        raise ValueError('duplicate methods in comparison')
    from .common import validate_training_method
    for cfg in configs:validate_training_method(cfg)
    reference=common_identity(configs[0])
    if any(common_identity(c)!=reference for c in configs[1:]):
        raise ValueError('non-allowlisted resolved configuration difference')
    staged=[c for c in configs if c["training_method"] in ("hybrid_insulated","two_stage")]
    if staged and any(stage1_identity(c)!=stage1_identity(staged[0]) for c in staged[1:]):
        raise ValueError("shared stage1 configuration difference")
    return reference

def stage1_identity(cfg):
    result=common_identity(cfg)
    result['stages']=[copy.deepcopy(s) for s in cfg['stages'] if s['name']=='stage1']
    return result


def canonical_prompts(tokenizer,cfg):
    from fluxvla.transforms.subtask_fast_prompts import PrepareSubtaskFastPrompts
    return PrepareSubtaskFastPrompts(tokenizer,state_dim=14,max_len=cfg['max_prompt_length'],
        model_vocab_size=257152,low_level_action_tail='',low_level_eos=False,low_level_segmented=True,
        low_level_sees_task=False, all_causal=False, low_level_prompt_format='state_subtask')
