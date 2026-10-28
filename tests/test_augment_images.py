import copy
import importlib.util
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_augment():
    path = ROOT / 'fluxvla/transforms/augment_images.py'
    name = 'soma_fluxvla_image_augment'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


aug = _load_augment()
AugmentImages = aug.AugmentImages
Config = aug.Config
ImageAugment = aug.ImageAugment
Parameters = aug.Parameters


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_temporal_consistency_rng_and_input_preservation():
    image = torch.randint(256, (3, 48, 64), dtype=torch.uint8)
    frames = image.repeat(3, 1, 1, 1)
    original = frames.clone()
    transform = ImageAugment()
    first = transform(frames, generator=torch.Generator().manual_seed(7))
    second = transform(frames, generator=torch.Generator().manual_seed(7))
    assert torch.equal(first, second)
    assert torch.equal(first[0], first[1])
    assert torch.equal(frames, original)
    assert first.shape == (3, 3, 48, 64)
    assert first.min() >= 0 and first.max() <= 1
    assert not torch.equal(
        first, transform(frames, generator=torch.Generator().manual_seed(8)))


def test_additive_brightness_constant_image():
    image = torch.tensor([0.1, 0.3, 0.5])[:, None, None].expand(3, 8, 8)
    params = Parameters(0.2, 1.2, 1, 0)
    result = ImageAugment()(image, parameters=params)
    assert torch.allclose(result, (image + 0.2).clamp(0, 1), atol=1e-6)


def test_evaluation_is_identity():
    image = torch.rand(3, 48, 64)
    transform = ImageAugment(training=False)
    assert torch.equal(transform(image), image)


def test_wrapper_cameras_independent_and_temporally_consistent():
    frame = np.random.default_rng(0).integers(0, 256, (3, 48, 64), dtype=np.uint8)
    images = [frame.copy() for _ in range(9)]
    original = copy.deepcopy(images)
    data = {
        'images': images,
        'actions': np.arange(14, dtype=np.float32),
        'states': np.ones(14, dtype=np.float32),
        'task_description': 'fold',
    }
    torch.manual_seed(11)
    result = AugmentImages(cameras=3)(data)
    assert result['actions'] is data['actions']
    assert result['states'] is data['states']
    assert result['task_description'] == 'fold'
    assert all(np.array_equal(a, b) for a, b in zip(original, images))
    out = result['images']
    assert len(out) == 9
    assert all(item.dtype == np.uint8 and item.shape == (3, 48, 64) for item in out)
    for camera in range(3):
        first = out[camera * 3]
        for offset in range(1, 3):
            np.testing.assert_array_equal(out[camera * 3 + offset], first)
    assert not np.array_equal(out[0], out[3])
    assert not np.array_equal(out[3], out[6])


def test_uint8_pipeline_survives_normalize_contract():
    images = [np.full((3, 32, 32), 180, dtype=np.uint8) for _ in range(3)]
    data = {'images': images}
    torch.manual_seed(3)
    data = AugmentImages(cameras=3)(data)
    stacked = np.stack(data['images'])
    assert stacked.dtype == np.uint8
    values = stacked.astype(np.float32) / 255.0 * 2.0 - 1.0
    assert values.min() >= -1 and values.max() <= 1
    assert values.mean() > -0.9
    assert stacked.shape == (3, 3, 32, 32)


def test_float_0_255_is_accepted_as_uint8():
    images = [np.full((3, 12, 12), 90.4, dtype=np.float32) for _ in range(3)]
    result = AugmentImages(training=False, cameras=3)({'images': images})
    assert all(item.dtype == np.uint8 and item.shape == (3, 12, 12) for item in result['images'])


def test_padding_images_stay_zero():
    images = [np.full((3, 24, 24), 200, dtype=np.uint8) for _ in range(3)]
    images.append(np.zeros((3, 24, 24), dtype=np.uint8))
    data = {'images': images, 'img_masks': np.array([True, True, True, False])}
    result = AugmentImages(cameras=3)(data)
    assert result['images'][-1].shape == (3, 24, 24)
    assert np.all(result['images'][-1] == 0)
    np.testing.assert_array_equal(result['img_masks'], data['img_masks'])
    assert not np.all(result['images'][0] == 0)


def test_reject_interleaved_padding():
    images = [np.zeros((3, 8, 8), dtype=np.uint8) for _ in range(4)]
    data = {'images': images, 'img_masks': np.array([True, False, True, True])}
    with pytest.raises(ValueError, match='trailing'):
        AugmentImages(cameras=3)(data)


def test_concatenated_layout_roundtrip():
    stacked = np.random.default_rng(2).integers(0, 256, (9, 20, 20), dtype=np.uint8)
    data = {'images': stacked}
    result = AugmentImages(cameras=3)(data)
    assert result['images'].shape == (9, 20, 20)
    assert result['images'].dtype == np.uint8


@pytest.mark.parametrize(
    'images',
    [
        [np.ones((3, 8, 8), dtype=np.float32) * 0.4 for _ in range(3)],
        [np.full((3, 8, 8), -0.5, dtype=np.float32) for _ in range(3)],
        [np.ones((8, 8, 3), dtype=np.uint8) for _ in range(3)],
        [np.full((3, 8, 8), np.nan, dtype=np.float32) for _ in range(3)],
    ],
)
def test_reject_invalid_wrapper_inputs(images):
    with pytest.raises(ValueError):
        AugmentImages(cameras=3)({'images': images})


def test_reject_non_divisible_camera_count():
    images = [np.zeros((3, 8, 8), dtype=np.uint8) for _ in range(2)]
    with pytest.raises(ValueError):
        AugmentImages(cameras=3)({'images': images})


def test_core_rejects_out_of_range():
    with pytest.raises(ValueError):
        ImageAugment()(torch.ones(3, 8, 8) * 2)
    with pytest.raises(ValueError):
        ImageAugment()(torch.full((3, 8, 8), float('nan')))


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        Config(brightness=-0.1)
    with pytest.raises(ValueError):
        Config(hue=0.6)
    with pytest.raises(ValueError):
        Config(contrast=(1.2, 0.8))


def _transform_types(config_path):
    config = runpy.run_path(str(config_path))
    return [
        item['type']
        for item in config['train_dataloader']['dataset']['datasets'][0]['transforms']
    ]


def test_aug_config_inserts_before_resize_and_has_no_crop_fields():
    baseline = _transform_types(ROOT / 'configs/pi05_piper.py')
    config = runpy.run_path(str(ROOT / 'configs/pi05_piper_aug.py'))
    items = config['train_dataloader']['dataset']['datasets'][0]['transforms']
    augmented = [item['type'] for item in items]
    aug_cfg = next(item for item in items if item['type'] == 'AugmentImages')
    assert 'crop_area' not in aug_cfg
    assert 'eval_center_crop' not in aug_cfg
    assert 'mode' not in aug_cfg
    assert 'size' not in aug_cfg
    assert 'AugmentImages' not in baseline
    assert 'AugmentImages' in augmented
    assert augmented.index('AugmentImages') < augmented.index('ResizeImages')
    assert augmented.index('ResizeImages') < augmented.index('SimpleNormalizeImages')
    assert baseline == [
        'ProcessParquetInputs',
        'ProcessLiberoActions',
        'NormalizeStatesAndActions',
        'PreparePromptWithState',
        'ProcessPrompts',
        'ResizeImages',
        'SimpleNormalizeImages',
    ]
    source = (ROOT / 'fluxvla/transforms/augment_images.py').read_text()
    for banned in (
        'crop_area', 'eval_center_crop', 'grid_sample', 'native_aspect',
        'openvla_square', 'center crop',
    ):
        assert banned not in source
    init_text = (ROOT / 'fluxvla/transforms/__init__.py').read_text()
    assert 'register_module(module=AugmentImages' in init_text
    assert 'soma_lerobot_augmentation' not in init_text
    assert 'soma-vla' not in source


def test_eval_wrapper_is_spatial_identity():
    images = [
        np.random.default_rng(4).integers(0, 256, (3, 32, 40), dtype=np.uint8)
        for _ in range(3)
    ]
    transform = AugmentImages(training=False, cameras=3)
    result = transform({'images': copy.deepcopy(images)})['images']
    for original, out in zip(images, result):
        np.testing.assert_array_equal(out, original)
