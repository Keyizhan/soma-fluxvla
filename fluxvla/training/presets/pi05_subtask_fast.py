"""FAST-token presets for the pi0.5 subtask + knowledge-insulation reproduction.

New controlled recipes use one valid-token cross entropy across subtask and FAST:
  stage1_fast: L_token
  hybrid / hybrid_insulated: L_token + lambda * L_flow (default lambda=1)
  action_expert / flow_only: L_flow
Subtask and FAST means remain separate diagnostics, never extra training weights.
Legacy separate-weight model configurations remain loadable explicitly.

state_dim=14 encodes only the real Piper dimensions: 32 bins cost 150 prefix tokens,
Only physical state dimensions are tokenized; action padding stays outside text.
"""
import copy
from pathlib import Path
import runpy

_BASE = copy.deepcopy(runpy.run_path(str(Path(__file__).with_name('pi05_piper.py')))['model'])

FAST_VOCAB_SIZE = 2048
FAST_SKIP_TOKENS = 128
MAX_TOKEN_LEN = 256
STATE_DIM = 14


def _model(token, action, stop_gradient=False):
    model = copy.deepcopy(_BASE)
    model.update(type='PI05Subtask', rtc_training_config=copy.deepcopy(_BASE['rtc_training_config']), ori_action_dim=14,
                 loss_reduction='local',
                 text_loss_mode='unified_token', token_loss_weight=token,
                 hl_loss_weight=0., fast_loss_weight=0., action_loss_weight=action,
                 fast_vocab_size=FAST_VOCAB_SIZE, fast_skip_tokens=FAST_SKIP_TOKENS,
                 # MUST inherit/use true Beta sampling for all subtask methods.
                 time_sampler='beta', time_beta_alpha=1.5, time_beta_beta=1.0,
                 rope_precision='jax', attn_math='fp32',
                 low_level_prefix_source='text', low_level_text_attention='bidirectional',
                 low_level_task_independent=True,
                 stop_gradient_flow_to_prefix=stop_gradient,
                 flow_prefix_cache=True, text_use_cache=True, action_prefix_cache=True,
                 # Decode in the training precision; precision changes require separate validation.
                 text_decode_fp32=False)
    return model


def _contract(name, token, action, stop_gradient):
    return dict(
        version='pi05-subtask-paper-v1', prompt_contract='pi05-subtask-paper-v1', preset=name,
        loss_objective='unified-token-ce-v1', text_loss_mode='unified_token',
        weights=dict(token=token, flow=action),
        text_reduction='sum valid subtask and FAST NLL / sum valid token weights',
        stop_gradient_flow_to_prefix=stop_gradient,
        high_condition='observation + overall_task',
        low_condition='observation + subtask, bidirectional flow prefix',
        text_mask='bidirectional conditioning prefix; causal high-level outputs', greedy=True,
        max_token_len=MAX_TOKEN_LEN, state_dim=STATE_DIM,
        fast_vocab_size=FAST_VOCAB_SIZE, fast_skip_tokens=FAST_SKIP_TOKENS,
        rtc_enabled=True, action_dimension=14, horizon=50,
        rope_precision="jax", attn_math="fp32",
        all_causal=False, low_level_sees_task=False,
        low_level_prompt_format='state_subtask', low_level_action_tail='',
        low_level_eos=False, low_level_segmented=True,
        low_level_prefix_source="text",
        conditioning=dict(training_action_prefix="text_bidirectional",
                          high_level_text_attention="full_prefix_causal_outputs",
                          generation_is_experimental=False),
        low_level_sees_fast_tokens=False,
        fast_is_training_time_supervision_only=True,
        synthetic_identity_is_semantic_evidence=False)


COLLATOR_KEYS = [
    'images', 'img_masks', 'states', 'actions', 'action_masks',
    'hl_input_ids', 'hl_valid_mask', 'hl_ar_mask', 'hl_labels',
    'hl_subtask_region_mask', 'hl_action_region_mask',
    'll_input_ids', 'll_valid_mask', 'has_hl', 'has_action']

# Default export: stage 1 (subtask + FAST, no flow) — the missing half of knowledge insulation.
model = _model(1.0, 0.0)
architecture_contract = _contract('stage1_fast', 1.0, 0.0, False)
collator = dict(type='DictCollator', keys=COLLATOR_KEYS,
                meta_keys=['overall_task', 'subtask'])

PRESETS = {
    'stage1_fast': dict(model=_model(1.0, 0.0), contract=_contract('stage1_fast', 1.0, 0.0, False)),
    'action_expert': dict(model=_model(0.0, 1.0, True), contract=_contract('action_expert', 0.0, 1.0, True)),
    'flow_only': dict(model=_model(0.0, 1.0), contract=_contract('flow_only', 0.0, 1.0, False)),
    'hybrid': dict(model=_model(1.0, 1.0), contract=_contract('hybrid', 1.0, 1.0, False)),
    'hybrid_insulated': dict(
        model=_model(1.0, 1.0, stop_gradient=True),
        contract=_contract('hybrid_insulated', 1.0, 1.0, True)),
}
