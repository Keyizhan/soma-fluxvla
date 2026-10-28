# Ordinary PI0.5 parquet training with online RGB colour jitter.
# Leaves the no-augmentation pi05_piper baseline unchanged.
# No spatial crop: train and serve both use the full frame.
# Do not use this recipe for subtask / bbox / controlled comparison runs.
import copy
import runpy

_BASE = {
    k: v for k, v in runpy.run_module(
        'fluxvla.training.presets.pi05_piper').items() if not k.startswith('_')
}
globals().update(_BASE)

_AUGMENT = {
    'type': 'AugmentImages',
    'training': True,
    'brightness': 0.2,
    'contrast': (0.8, 1.2),
    'saturation': (0.8, 1.2),
    'hue': 0.05,
    'cameras': 3,
}

train_dataloader = copy.deepcopy(_BASE['train_dataloader'])
_transforms = train_dataloader['dataset']['datasets'][0]['transforms']
if any(item.get('type') == 'AugmentImages' for item in _transforms):
    raise ValueError('AugmentImages is already present in the baseline recipe')
_insert = next(
    (i for i, item in enumerate(_transforms) if item.get('type') == 'ResizeImages'),
    None,
)
if _insert is None:
    raise ValueError('Expected ResizeImages in the ordinary PI0.5 transform list')
_transforms.insert(_insert, _AUGMENT)

runner = copy.deepcopy(_BASE['runner'])
runner['metric']['run_dir'] = 'work_dirs/pi05_piper_aug'
