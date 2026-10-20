"""FAST-token presets for the pi0.5 subtask + knowledge-insulation reproduction.

Weights mirror the reference presets (Ke-Wang1017/openpi_subtask @385d0e8):
  stage1_fast  10 / 1 / 0    VLM learns an action representation (subtask + FAST CE)
  flow_only     0 / 0 / 1    flow-matching baseline (our earlier runs)
  hybrid     0.15 / 0.15 / 1 final recipe (subtask + FAST + flow)
`hl` = subtask CE, `fast` = FAST action-token CE, `action` = flow-matching MSE.

state_dim=14 encodes only the real Piper dimensions: 32 bins cost 150 prefix tokens,
14 bins cost 73 (see experiments/pi05_subtask_architecture/fast_stage1_20261017/REPORT.md).
"""
import copy
from pathlib import Path
import runpy

_BASE = copy.deepcopy(runpy.run_path(str(Path(__file__).with_name('pi05_piper.py')))['model'])

FAST_VOCAB_SIZE = 2048
FAST_SKIP_TOKENS = 128
MAX_TOKEN_LEN = 256
STATE_DIM = 14


def _model(hl, fast, action, stop_gradient=False):
    model = copy.deepcopy(_BASE)
    model.update(type='PI05Subtask', rtc_training_config=None, ori_action_dim=14,
                 hl_loss_weight=hl, fast_loss_weight=fast, action_loss_weight=action,
                 fast_vocab_size=FAST_VOCAB_SIZE, fast_skip_tokens=FAST_SKIP_TOKENS,
                 exact_beta_time=True,
                 rope_precision='jax', attn_math='fp32',
                 low_level_prefix_source='text',
                 stop_gradient_flow_to_prefix=stop_gradient,
                 flow_prefix_cache=True, text_use_cache=True, action_prefix_cache=True,
                 # Must stay False: training runs under bf16 autocast, so decoding the text
                 # branch in fp32 produces a DIFFERENT function. Measured against the
                 # training-style forward: cache+bf16 = 1.25 logits / 0 flips,
                 # cache+fp32 = 3.1, recompute+fp32 = 32.8 / 10 flips.
                 text_decode_fp32=False)
    return model


def _contract(name, hl, fast, action, stop_gradient):
    return dict(
        version='pi05-subtask-fast-v1', preset=name,
        weights=dict(subtask=hl, fast=fast, flow=action),
        stop_gradient_flow_to_prefix=stop_gradient,
        high_condition='observation + overall_task',
        low_condition='observation + overall_task + subtask, bidirectional flow prefix',
        text_mask='high-level text causal; low-level flow prefix bidirectional', greedy=True,
        max_token_len=MAX_TOKEN_LEN, state_dim=STATE_DIM,
        fast_vocab_size=FAST_VOCAB_SIZE, fast_skip_tokens=FAST_SKIP_TOKENS,
        rtc_enabled=False, action_dimension=14, horizon=50,
        rope_precision="jax", attn_math="fp32",
        all_causal=True, low_level_sees_task=True,
        low_level_prefix_source="text",
        conditioning=dict(training_action_prefix="text_bidirectional",
                          high_level_text_attention="causal",
                          generation_is_experimental=True),
        low_level_sees_fast_tokens=False,
        fast_is_training_time_supervision_only=True,
        synthetic_identity_is_semantic_evidence=False)


COLLATOR_KEYS = [
    'images', 'img_masks', 'states', 'actions', 'action_masks',
    'hl_input_ids', 'hl_valid_mask', 'hl_ar_mask', 'hl_labels',
    'hl_subtask_region_mask', 'hl_action_region_mask',
    'll_input_ids', 'll_valid_mask', 'has_hl', 'has_action']

# Default export: stage 1 (subtask + FAST, no flow) — the missing half of knowledge insulation.
model = _model(10.0, 1.0, 0.0)
architecture_contract = _contract('stage1_fast', 10.0, 1.0, 0.0, False)
collator = dict(type='DictCollator', keys=COLLATOR_KEYS,
                meta_keys=['overall_task', 'subtask'])

PRESETS = {
    'stage1_fast': dict(model=_model(10.0, 1.0, 0.0), contract=_contract('stage1_fast', 10.0, 1.0, 0.0, False)),
    'flow_only': dict(model=_model(0.0, 0.0, 1.0), contract=_contract('flow_only', 0.0, 0.0, 1.0, False)),
    'hybrid': dict(model=_model(0.15, 0.15, 1.0), contract=_contract('hybrid', 0.15, 0.15, 1.0, False)),
    'hybrid_insulated': dict(
        model=_model(0.15, 0.15, 1.0, stop_gradient=True),
        contract=_contract('hybrid_insulated', 0.15, 0.15, 1.0, True)),
}
