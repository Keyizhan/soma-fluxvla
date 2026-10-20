# Derived from FluxVLA (Limx Dynamics); see LICENSE and NOTICE.
"""
This module defines the abstract interface and default
implementation for Vision Backbones
(visual feature extractors). It includes:

- `VisionBackbone`: An abstract base class for visual featurizers.
- `TimmViTBackbone`: A base class for feature extractors built
    using TIMM Vision Transformer (ViT) models.
- Supporting transform utilities such as `LetterboxPad` and
    `unpack_tuple`.

These classes are designed to be used as plug-and-play modules in
multi-modal systems such as Visual Language Models (VLMs) or image
captioning architectures.
"""
import einops
import numpy as np

def pad_to_dim(x: np.ndarray, target_dim: int, axis: int=-1) -> np.ndarray:
    """Pad an array to the target dimension with zeros
        along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width)
    return x

def parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, 'c h w -> h w c')
    return image
