# Derived from FluxVLA (Limx Dynamics); see LICENSE and NOTICE.
"""Online RGB colour jitter for the ordinary PI0.5 parquet pipeline.

Adapted from OpenVLA-style TensorFlow colour operators (additive brightness,
per-channel contrast, HSV saturation, hue). There is no spatial crop: train
and serve both use the full frame. This module is self-contained; it does not
import other checkouts or write datasets. Insert before SimpleNormalizeImages.
Output is uint8 [0, 255] so the existing x/255*2-1 normalisation remains valid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import functional as F


def _pair(value: Sequence[float]) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('Expected an ordered pair')
    return (float(value[0]), float(value[1]))


@dataclass(frozen=True)
class Config:
    brightness: float = 0.2  # additive offset, not a torchvision factor
    contrast: tuple[float, float] = (0.8, 1.2)
    saturation: tuple[float, float] = (0.8, 1.2)
    hue: float = 0.05

    def __post_init__(self):
        object.__setattr__(self, 'contrast', _pair(self.contrast))
        object.__setattr__(self, 'saturation', _pair(self.saturation))
        if not 0 <= self.brightness <= 1 or not 0 <= self.hue <= 0.5:
            raise ValueError('Invalid brightness/hue bounds')
        for bounds in (self.contrast, self.saturation):
            if not 0 <= bounds[0] <= bounds[1] < math.inf:
                raise ValueError('Factors must be finite ordered nonnegative pairs')


@dataclass(frozen=True)
class Parameters:
    brightness: float
    contrast: float
    saturation: float
    hue: float


class ImageAugment:
    """CPU RGB CHW or TCHW -> float32 [0, 1], sharing parameters across T.

    Spatial size is unchanged. Call once per camera window.
    Not bitwise-equivalent to TensorFlow.
    """

    def __init__(self, config: Config | None = None, *, training: bool = True):
        self.config = config if config is not None else Config()
        self.training = training

    def sample(self, generator: torch.Generator | None = None) -> Parameters:
        cfg = self.config
        if not self.training:
            return Parameters(0, 1, 1, 0)
        random = torch.rand(4, generator=generator).tolist()
        return Parameters(
            (2 * random[0] - 1) * cfg.brightness,
            cfg.contrast[0] + random[1] * (cfg.contrast[1] - cfg.contrast[0]),
            cfg.saturation[0] + random[2] * (cfg.saturation[1] - cfg.saturation[0]),
            (2 * random[3] - 1) * cfg.hue,
        )

    def __call__(
        self,
        image: Tensor,
        *,
        generator: torch.Generator | None = None,
        parameters: Parameters | None = None,
    ) -> Tensor:
        if not isinstance(image, Tensor) or image.ndim not in (3, 4) or image.shape[-3] != 3:
            raise ValueError('Expected RGB CHW or TCHW torch tensor')
        if image.device.type != 'cpu':
            raise ValueError('Apply this adapter on CPU before transfer to the model')
        if image.dtype == torch.uint8:
            x = image.float() / 255
        elif image.is_floating_point():
            x = image.float()
        else:
            raise ValueError('Expected uint8 or floating point RGB')
        if not x.numel() or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError(
                'Float input must be finite, nonempty and in [0,1], before Normalize')
        single = x.ndim == 3
        if single:
            x = x.unsqueeze(0)
        p = parameters if parameters is not None else self.sample(generator)
        if p.brightness != 0:
            x = (x + p.brightness).clamp(0, 1)
        if p.contrast != 1:
            mean = x.mean(dim=(-2, -1), keepdim=True)
            x = ((x - mean) * p.contrast + mean).clamp(0, 1)
        if p.saturation != 1:
            maximum = x.amax(dim=-3, keepdim=True)
            delta = maximum - x.amin(dim=-3, keepdim=True)
            factor = torch.minimum(
                torch.full_like(delta, p.saturation),
                maximum / delta.clamp_min(1e-12),
            )
            x = (maximum - (maximum - x) * factor).clamp(0, 1)
        if p.hue != 0:
            x = F.adjust_hue(x, p.hue).clamp(0, 1)
        return x[0] if single else x


def _to_uint8(image: Tensor) -> Tensor:
    return (image.clamp(0, 1) * 255).round().to(torch.uint8)


def _as_nchw(images) -> tuple[np.ndarray, str]:
    if isinstance(images, (list, tuple)):
        if not images:
            raise ValueError('images must be nonempty')
        frames = [np.asarray(image) for image in images]
        stacked = np.stack(frames, axis=0)
        return _validate_nchw(stacked), 'list'
    array = np.asarray(images)
    if array.ndim == 4:
        return _validate_nchw(array), 'nchw'
    if array.ndim == 3:
        if array.shape[0] % 3 != 0:
            raise ValueError('Concatenated images must have a channel count divisible by 3')
        stacked = array.reshape(-1, 3, array.shape[-2], array.shape[-1])
        return _validate_nchw(stacked), 'concat'
    raise ValueError('Expected a list of CHW images or a (3N,H,W)/(N,3,H,W) array')


def _validate_nchw(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError('Expected RGB CHW frames')
    if frames.size == 0:
        raise ValueError('images must be nonempty')
    if np.issubdtype(frames.dtype, np.floating):
        if not np.isfinite(frames).all():
            raise ValueError('Float images must be finite')
        if frames.min() < 0:
            raise ValueError('Rejected already-normalised images (min < 0)')
        if frames.max() <= 1.0:
            raise ValueError(
                'Rejected [0,1] float images; AugmentImages must run on uint8 '
                '[0,255] before SimpleNormalizeImages')
        if frames.max() > 255:
            raise ValueError('Float images must be in [0, 255]')
        return np.rint(frames).astype(np.uint8)
    if frames.dtype != np.uint8:
        raise ValueError('Expected uint8 or floating point RGB')
    return frames


def _real_mask(data: dict, count: int) -> np.ndarray:
    if 'img_masks' not in data:
        return np.ones(count, dtype=bool)
    masks = np.asarray(data['img_masks']).astype(bool).reshape(-1)
    if masks.shape != (count,):
        raise ValueError('img_masks length must match the number of images')
    return masks


class AugmentImages:
    """Per-camera online RGB colour jitter for ProcessParquetInputs outputs.

    Camera-major layout: [cam0_t0..cam0_tT, cam1_..., ...], matching
    ProcessParquetInputs. History frames of one camera share parameters.
    Spatial size is preserved. Padding images with img_masks=False stay zero
    and are not jittered.
    """

    def __init__(
        self,
        training: bool = True,
        brightness: float = 0.2,
        contrast: Sequence[float] = (0.8, 1.2),
        saturation: Sequence[float] = (0.8, 1.2),
        hue: float = 0.05,
        cameras: int = 3,
        *args,
        **kwargs,
    ):
        if int(cameras) < 1:
            raise ValueError('cameras must be a positive integer')
        self.cameras = int(cameras)
        self.augment = ImageAugment(
            Config(
                brightness=float(brightness),
                contrast=_pair(contrast),
                saturation=_pair(saturation),
                hue=float(hue),
            ),
            training=bool(training),
        )

    def __call__(self, data: dict):
        if 'images' not in data:
            raise KeyError("Input data must contain 'images' key")
        frames, layout = _as_nchw(data['images'])
        masks = _real_mask(data, frames.shape[0])
        if not masks.all():
            first_pad = int(np.argmax(~masks))
            if np.any(masks[first_pad:]):
                raise ValueError('img_masks padding must be trailing')
        real = int(masks.sum())
        if real == 0:
            raise ValueError('Expected at least one unmasked image')
        if real % self.cameras != 0:
            raise ValueError(
                f'Real image count {real} is not divisible by cameras={self.cameras}')
        timesteps = real // self.cameras
        height, width = frames.shape[-2:]
        output = np.zeros((frames.shape[0], 3, height, width), dtype=np.uint8)
        real_frames = frames[masks]
        grouped = real_frames.reshape(self.cameras, timesteps, 3, height, width)
        for camera in range(self.cameras):
            window = torch.from_numpy(np.ascontiguousarray(grouped[camera]))
            augmented = _to_uint8(self.augment(window)).numpy()
            output[np.flatnonzero(masks)[camera * timesteps:(camera + 1) * timesteps]] = (
                augmented
            )
        if layout == 'list':
            data['images'] = [frame for frame in output]
        elif layout == 'nchw':
            data['images'] = output
        else:
            data['images'] = output.reshape(-1, height, width)
        return data


def _register():
    try:
        from fluxvla.engines import TRANSFORMS
    except ImportError:
        return
    TRANSFORMS.register_module(module=AugmentImages, force=True)


_register()
