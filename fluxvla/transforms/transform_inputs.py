# Derived from FluxVLA (Limx Dynamics); see LICENSE and NOTICE.
import logging
import os
from pathlib import Path
from typing import Dict, List
import av
import numpy as np
import torch
import torchvision
from fluxvla.datasets.utils.video_decode import build_lerobot_video_path, decode_video_frames_torchvision
from fluxvla.engines import TRANSFORMS

@TRANSFORMS.register_module()
class ProcessParquetInputs:
    """Process inputs for Parquet dataset.
    This transform processes the inputs from the Parquet
    dataset to match the expected format for the model.
    It pads the state and action dimensions to the specified
    action dimension and parses the images from the input data.
    The processed inputs are returned in a dictionary format
    that includes the state, images, image masks, and
    actions (if available). The prompt is also included
    if it exists in the input data.

    Args:
        parquet_keys (List[str]): List of keys to extract
            from the parquet data.
        video_keys (List[str]): List of keys corresponding
            to video data.
        data_root (str): Root directory for the video files.
        name_mappings (Dict, optional): Optional dictionary
            to map original keys to new keys.
            Defaults to None.
    """

    def __init__(self, parquet_keys: List[str], video_keys: List[str], name_mappings: Dict=None, embodiment_id: int=None, embodiment_dim: int=None, num_padding_imgs: int=0, dataset_name: str=None, video_cache: dict=None):
        self.parquet_keys = parquet_keys
        self.video_keys = video_keys
        self.name_mappings = name_mappings
        self.embodiment_id = embodiment_id
        self.embodiment_dim = embodiment_dim
        self.num_padding_imgs = num_padding_imgs
        self.dataset_name = dataset_name
        self.video_cache_config = video_cache
        self._video_cache = None

    def decode_video_frames_torchvision(self, video_path: Path | str, timestamps: list[float], tolerance_s: float, backend: str='pyav', log_loaded_timestamps: bool=False) -> torch.Tensor:
        """ Decode video frames using torchvision.

        Args:
            video_path (Path | str): Path to the video file.
            timestamps (list[float]): List of timestamps to decode.
            tolerance_s (float): Tolerance in seconds for the timestamps.
            backend (str): Backend to use for video decoding.
                Defaults to 'pyav'.
            log_loaded_timestamps (bool): Whether to log the loaded timestamps.
                Defaults to False.

        Returns:
            torch.Tensor: Tensor of decoded video frames.
        """
        if self.video_cache_config is not None:
            if self._video_cache is None:
                from fluxvla.datasets.cache import VideoCache
                self._video_cache = VideoCache(**self.video_cache_config)
            return torch.from_numpy(self._video_cache.read(video_path, timestamps, tolerance_s))
        return decode_video_frames_torchvision(
            video_path, timestamps, tolerance_s=tolerance_s, backend=backend,
            log_loaded_timestamps=log_loaded_timestamps)

    def __call__(self, data):
        assert 'info' in data, "Input data must contain 'info' key"
        info = data['info']
        inputs = dict()
        assert 'video_path' in info, "Input data must contain 'video_path' key"
        for key in self.parquet_keys:
            try:
                value = data[key]
            except KeyError as exc:
                raise KeyError(f'Missing input data key: {key}') from exc
            mapped_names = None
            if self.name_mappings is not None:
                mapped_names = self.name_mappings.get(key)
            if mapped_names is not None:
                if isinstance(mapped_names, str):
                    if isinstance(value, list) or isinstance(value, float):
                        inputs[mapped_names] = np.array(value)
                    else:
                        inputs[mapped_names] = value
                else:
                    for mapped_key in mapped_names:
                        if isinstance(value, list) or isinstance(value, float):
                            inputs[mapped_key] = np.array(value)
                        else:
                            inputs[mapped_key] = value
            elif isinstance(value, list) or isinstance(value, float):
                inputs[key] = np.array(value)
            else:
                inputs[key] = value
        images = list()
        img_masks = list()
        timestamps = data.get('frame_timestamps', [data['timestamp']])
        for video_key in self.video_keys:
            video_path = build_lerobot_video_path(data['data_root'], data['info'], data, int(data['episode_index']), video_key)
            assert os.path.exists(video_path), f'Video file not found: {video_path}'
            unique_ts = sorted(set(timestamps))
            frames_tensor = self.decode_video_frames_torchvision(video_path, unique_ts, 0.1)
            ts_to_frame = {ts: frames_tensor[i] for i, ts in enumerate(unique_ts)}
            for ts in timestamps:
                nearest = min(unique_ts, key=lambda x: abs(x - ts))
                images.append(ts_to_frame[nearest].numpy())
            for _ in timestamps:
                img_masks.append(True)
        if self.num_padding_imgs > 0 and len(images) > 0:
            padding_img = np.zeros_like(images[0])
            for _ in range(self.num_padding_imgs):
                images.append(padding_img)
                img_masks.append(False)
        inputs['images'] = images
        inputs['img_masks'] = np.array(img_masks)
        inputs['task_description'] = data.get('task_description', '')
        if self.dataset_name is not None:
            inputs['dataset_name'] = self.dataset_name
        if self.embodiment_id is not None:
            inputs['embodiment_ids'] = np.array(self.embodiment_id)
        if 'frame_masks' in data:
            inputs['frame_masks'] = data['frame_masks']
        if 'sample_weight' in data:
            inputs['sample_weight'] = np.asarray(data['sample_weight'], dtype=np.float32)
        return inputs

    def read_video_frame(self, video_path: str, frame_idx: int):
        container = av.open(video_path)
        for i, frame in enumerate(container.decode(video=0)):
            if i == frame_idx:
                return frame.to_ndarray(format='rgb24')
