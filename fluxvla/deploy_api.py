"""Public interface soma-deploy uses (TODO D13). Changing any name or removing a
parameter here is a breaking change; ``tests/test_deploy_api_contract.py`` pins it.

Names resolve lazily so importing this module does not pull in torch.
"""
import importlib

_EXPORTS = {
    'build_vla_from_cfg': 'fluxvla.engines.utils.builder',
    'build_dataset_from_cfg': 'fluxvla.engines.utils.builder',
    'build_tokenizer_from_cfg': 'fluxvla.engines.utils.builder',
    'TRANSFORMS': 'fluxvla.engines.utils.root',
    'str_to_dtype': 'fluxvla.engines.utils.name_map',
    'set_seed_everywhere': 'fluxvla.engines.utils.torch_utils',
    'PI05Subtask': 'fluxvla.models.vlas.pi05_subtask',
    'PrepareSubtaskFastPrompts': 'fluxvla.transforms.subtask_fast_prompts',
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'fluxvla.deploy_api has no attribute {name!r}')
    return getattr(importlib.import_module(_EXPORTS[name]), name)
