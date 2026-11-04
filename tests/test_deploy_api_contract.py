"""soma-deploy contract (D13): names exist where deploy_api says, and the pinned
parameters keep their names and order. New parameters may be appended only with
defaults. Static (AST) so it runs without the training stack."""
import ast
from pathlib import Path

from fluxvla import deploy_api

ROOT = Path(__file__).resolve().parents[1]

PINNED = {
    'build_vla_from_cfg': ['cfg', 'default_args'],
    'build_dataset_from_cfg': ['cfg', 'default_args'],
    'build_tokenizer_from_cfg': ['cfg', 'default_args'],
    'str_to_dtype': ['s'],
    'set_seed_everywhere': ['seed'],
    'PI05Subtask': ['hl_loss_weight', 'action_loss_weight', 'fast_loss_weight', 'text_loss_mode',
                    'token_loss_weight', 'split_subtask_loss', 'subtask_key_weight',
                    'subtask_template_weight', 'normalize_text_weights', 'bbox', 'fast_vocab_size',
                    'fast_skip_tokens', 'exact_beta_time', 'rope_precision', 'attn_math',
                    'warn_on_bf16_cache', 'low_level_prefix_source', 'low_level_task_independent',
                    'stop_gradient_flow_to_prefix', 'text_use_cache', 'action_prefix_cache',
                    'flow_prefix_cache', 'text_decode_fp32', 'low_level_text_attention',
                    'loss_reduction'],
    'PrepareSubtaskFastPrompts': ['tokenizer', 'max_len', 'state_dim', 'state_bins',
                                  'model_vocab_size', 'fast_vocab_size', 'fast_skip_tokens',
                                  'low_level_sees_task', 'all_causal', 'low_level_action_tail',
                                  'low_level_eos', 'low_level_segmented', 'low_level_prompt_format',
                                  'subtask_mask_spec', 'subtask_templates'],
}


def _definition(name):
    module = deploy_api._EXPORTS[name]
    tree = ast.parse((ROOT / (module.replace('.', '/') + '.py')).read_text())
    for node in tree.body:
        if getattr(node, 'name', None) == name:
            return node
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == name for t in node.targets):
            return node
    raise AssertionError(f'{name} not defined in {module}')


def _params(node):
    if isinstance(node, ast.ClassDef):
        node = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    args = node.args
    names = [a.arg for a in args.posonlyargs + args.args if a.arg != 'self']
    n_defaults = len(args.defaults)
    required = names[:len(names) - n_defaults] if n_defaults else names
    return names, required


def test_exports_are_pinned():
    assert set(deploy_api.__all__) == set(PINNED) | {'TRANSFORMS'}


def test_every_name_is_defined():
    for name in deploy_api.__all__:
        _definition(name)


def test_pinned_parameters_unchanged():
    for name, pinned in PINNED.items():
        names, required = _params(_definition(name))
        assert names[:len(pinned)] == pinned, f'{name}: parameters changed: {names}'
        extra_required = [p for p in names[len(pinned):] if p in required]
        assert not extra_required, f'{name}: new required parameters {extra_required}'


def test_import_is_lazy():
    import subprocess
    import sys
    code = 'import sys, fluxvla.deploy_api; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], check=True, cwd=ROOT)
