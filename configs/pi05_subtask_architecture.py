"""AI-designed architecture probe, separate from robot-task training configs."""
import copy
from pathlib import Path
import runpy

model = copy.deepcopy(runpy.run_path(str(Path(__file__).with_name('pi05_piper.py')))['model'])
model.update(type='PI05Subtask', rtc_training_config=None, ori_action_dim=32,
             hl_loss_weight=1.0, action_loss_weight=1.0,
             stop_gradient_flow_to_prefix=False, flow_prefix_cache=True,
             text_use_cache=True, action_prefix_cache=True, text_decode_fp32=True)

architecture_contract = dict(
    version='pi05-subtask-v2-kv', high_condition='observation + overall_task',
    low_condition='observation + subtask', two_separate_contexts=True,
    greedy=True, text_use_cache=True, action_prefix_cache=True, flow_prefix_cache=True,
    stop_gradient_flow_to_prefix=False, max_prompt_length=200,
    text_decode_fp32=True,
    rtc_enabled=False, action_dimension=32, horizon=50,
    synthetic_identity_is_semantic_evidence=False)

collator = dict(type='DictCollator', keys=[
    'images', 'img_masks', 'states', 'actions', 'action_masks',
    'hl_input_ids', 'hl_valid_mask', 'hl_ar_mask', 'hl_labels',
    'll_input_ids', 'll_valid_mask', 'has_hl', 'has_action'],
    meta_keys=['overall_task', 'subtask'])
