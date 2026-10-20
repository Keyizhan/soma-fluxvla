# Derived from FluxVLA (Limx Dynamics); see LICENSE and NOTICE.
import cv2
import numpy as np
from fluxvla.engines import TRANSFORMS
PAD_POSITIONS = ('top-left', 'top-right', 'bottom-left', 'bottom-right', 'center')
PAD_POSITIONS_TEXT = ', '.join(PAD_POSITIONS)

@TRANSFORMS.register_module()
class ResizeImages:
    """Resize images in the dataset to a specified
    height and width. This transform resizes all images
    in the 'image' dictionary of the input data
    to the specified dimensions.

    Args:
        height (int): The target height for the images.
        width (int): The target width for the images.
        preserve_leading_dims (bool): If True, treat the last three
            dimensions as CHW and preserve all leading dimensions.
    """

    def __init__(self, height, width, preserve_leading_dims: bool=False, *args, **kwargs):
        self.height = height
        self.width = width
        self.preserve_leading_dims = preserve_leading_dims

    def _resize_single_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image.transpose(1, 2, 0), (self.width, self.height), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)

    def _resize_preserve_leading_dims(self, images: np.ndarray) -> np.ndarray:
        original_shape = images.shape
        if images.ndim < 4:
            raise ValueError('Input image sequence must have at least 4 dimensions')
        flat_images = images.reshape(-1, original_shape[-3], original_shape[-2], original_shape[-1])
        resized_images = [self._resize_single_image(image) for image in flat_images]
        return np.stack(resized_images, axis=0).reshape(*original_shape[:-2], self.height, self.width)

    def __call__(self, data: dict):
        assert 'images' in data, "Input data must contain 'images' key"
        if self.preserve_leading_dims:
            data['images'] = self._resize_preserve_leading_dims(np.asarray(data['images']))
            return data
        if isinstance(data['images'], np.ndarray):
            assert data['images'].ndim == 3, "Input 'images' must be a 4D numpy array"
            images = data['images'].reshape(-1, 3, data['images'].shape[-2], data['images'].shape[-1])
        else:
            images = data['images']
        resized_images = list()
        for image in images:
            resized_images.append(self._resize_single_image(image))
        resized_images = np.concatenate(resized_images, axis=0)
        data['images'] = resized_images
        return data

@TRANSFORMS.register_module()
class SimpleNormalizeImages:
    """Simple normalization of images in the dataset.
    This transform normalizes all images in the 'images' dictionary
    by dividing by 255 and then mapping to the range [-1, 1].

    Args:
        None: This transform does not require any parameters.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, data: dict):
        assert 'images' in data, "Input data must contain 'images' key"
        images = data['images'].reshape(-1, 3, data['images'].shape[-2], data['images'].shape[-1])
        normalized_images = list()
        for image in images:
            normalized_image = image / 255.0 * 2.0 - 1.0
            normalized_images.append(normalized_image)
        normalized_images = np.concatenate(normalized_images, axis=0)
        data['images'] = normalized_images
        return data
