# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import shlex
from collections import deque
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import concatenate_datasets, load_dataset
from fluxvla.engines import DATASETS, build_transform_from_cfg


@DATASETS.register_module()
class ParquetDataset(Dataset):
    VERSION_FILE = 'meta/fluxvla_dataset_version.json'
    HF_REPO_ID = 'limxdynamics/FluxVLAData'
    HF_REVISION = 'main'

    def __init__(self,
                 data_root_path: Union[str, List[str]],
                 transforms: List[Dict],
                 action_window_size: int = 9,
                 action_key: str = 'observation.state',
                 use_delta: bool = False,
                 statistic_name: str = 'private',
                 window_start_idx: int = 1,
                 frame_window_size: int = 1,
                 frame_sample_stride: int = 1,
                 episode_indices: Optional[List[int]] = None,
                 episode_frame_ranges: Optional[Dict[int, List[int]]] = None,
                 train_episode_fraction: float = 1.0,
                 repeat_to_full_length: bool = False,
                 expose_index: bool = False,
                 expected_dataset_version: Optional[str] = None) -> None:
        """Initialize the Parquet dataset.

        Args:
            data_root_path (Union[str, List[str]]): Path(s) to the root
                directory(ies). The metadata will be loaded from
                `data_root_path/meta` and data from `data_root_path/data`.
                If a list is provided, multiple datasets will be loaded and
                concatenated.
            transforms (List[Dict]): List of transformation configurations.
            batch_transform (Union[dict, ConfigDict, Config]):
                Configuration for the batch transformation.
            episodes (list[int]): List of episode indices to include
                in the dataset.
            local_files_only (bool): Whether to use local files only.
            action_horizon (int): The number of time steps for the
                action sequence.
            video_backend (str, optional): Backend for
                video processing.
                Defaults to None.
            action_key (str): Key for the action data.
            use_delta (bool): Whether to use delta actions.
                Defaults to False.
            statistic_name (str): Name for the statistics collection.
                Defaults to 'private'.
            window_start_idx (int): Start index for the action window.
                Defaults to 1.
            frame_window_size (int): Number of video frames to expose via
                ``frame_timestamps``. Defaults to 1 (single current frame).
            frame_sample_stride (int): Stride (in dataset rows) between the
                sampled video frames. Defaults to 1 (consecutive frames).
                Increase this when the sampled frames should span a longer
                temporal window:
                ``(frame_window_size - 1) * frame_sample_stride`` rows.
            episode_indices (list[int], optional): Explicit immutable episode
                membership. It is mutually exclusive with fractional splits.
            episode_frame_ranges (dict[int, list[int]], optional): Optional
                end-exclusive ``[start, end]`` frame-index range for each
                selected episode. Intended for audited diagnostic subsets.
            train_episode_fraction (float): Fraction of episodes to sample
                from each data root, preserving original episode order.
                Defaults to 1.0.
            repeat_to_full_length (bool): If True, repeat the selected
                episode subset so `__len__` remains the full dataset length.
                This keeps epoch length based on full statistics while
                sampling only the selected train episode fraction.
            expose_index (bool): Whether to add the concatenated dataset index
                to each raw sample before transforms. This is useful for
                offline sample-weight transforms such as SARM RA-BC.
                Defaults to False.
            expected_dataset_version (str, optional): Expected FluxVLA dataset
                content version. If omitted, no version check is performed so
                existing local datasets remain usable.
        """
        super().__init__()
        if not 0 < train_episode_fraction <= 1:
            raise ValueError('train_episode_fraction must be in (0, 1].')
        if episode_indices is not None and train_episode_fraction != 1.0:
            raise ValueError('episode_indices and train_episode_fraction are '
                             'mutually exclusive.')
        self.action_window_size = action_window_size
        if isinstance(data_root_path, str):
            data_root_path = [data_root_path]
        self.data_root_path = data_root_path

        meta_root = [os.path.join(path, 'meta') for path in data_root_path]
        data_root = [os.path.join(path, 'data') for path in data_root_path]

        # Merge multiple meta_root
        all_stats = []
        all_tasks = []
        all_episodes = []
        info_list = []

        for dataset_root, root in zip(data_root_path, meta_root):
            info_path = os.path.join(root, 'info.json')
            assert os.path.exists(info_path), \
                f'Metadata file not found at {info_path}'
            with open(os.path.join(root, 'info.json'), 'rb') as f:
                info_list.append(json.load(f))
            if expected_dataset_version is not None:
                self._verify_dataset_version(
                    dataset_root=dataset_root,
                    expected_dataset_version=expected_dataset_version)

            stats_path = os.path.join(root, 'episodes_stats.jsonl')
            assert os.path.exists(stats_path), \
                f'Statistics file not found at {stats_path}'
            with open(
                    os.path.join(root, 'episodes_stats.jsonl'),
                    'r',
                    encoding='utf-8') as f:
                all_stats.extend([json.loads(line) for line in f])

            tasks_path = os.path.join(root, 'tasks.jsonl')
            assert os.path.exists(tasks_path), \
                f'Tasks file not found at {tasks_path}'
            with open(tasks_path, 'r', encoding='utf-8') as f:
                all_tasks.append([json.loads(line) for line in f])

            episodes_path = os.path.join(root, 'episodes.jsonl')
            assert os.path.exists(episodes_path), \
                f'Episodes file not found at {episodes_path}'
            with open(episodes_path, 'r', encoding='utf-8') as f:
                all_episodes.extend([json.loads(line) for line in f])

        self.info = info_list
        self.stats = all_stats
        self.tasks = all_tasks
        self.episodes = all_episodes
        # Summarize all data_root
        datasets = []
        dataset_sizes = []  # Record the size of each dataset
        for root in data_root:
            hf_dataset = load_dataset('parquet', data_dir=root, split='train')
            dataset_sizes.append(len(hf_dataset))
            datasets.append(hf_dataset)
        hf_dataset = concatenate_datasets(datasets)
        # Compute cumulative sizes for fast index lookup
        self.dataset_cumulative_sizes = np.cumsum([0] + dataset_sizes)
        self.dataset = hf_dataset
        self.full_length = len(self.dataset)
        self.sample_indices = self._build_sample_indices(
            train_episode_fraction, episode_indices, episode_frame_ranges)
        self.effective_length = (
            self.full_length
            if repeat_to_full_length else len(self.sample_indices))
        self.transforms = list()
        self.action_key = action_key
        self.use_delta = use_delta
        self.statistic_name = statistic_name
        self.window_start_idx = window_start_idx
        self.frame_window_size = frame_window_size
        self.frame_sample_stride = frame_sample_stride
        self.expose_index = expose_index
        for transform in transforms:
            self.transforms.append(build_transform_from_cfg(transform))

    @staticmethod
    def _read_dataset_version(path: str) -> Optional[str]:
        with open(path, 'r', encoding='utf-8') as f:
            raw_version = f.read().strip()
        if not raw_version:
            return None

        try:
            version_data = json.loads(raw_version)
        except json.JSONDecodeError:
            return raw_version

        if isinstance(version_data, dict):
            version = version_data.get('fluxvla_dataset_version',
                                       version_data.get('version'))
            return str(version) if version is not None else None
        if isinstance(version_data, str):
            return version_data
        return str(version_data)

    @classmethod
    def _dataset_refresh_command(cls, dataset_root: str) -> str:
        normalized_root = dataset_root.rstrip(os.sep)
        local_dir = os.path.dirname(normalized_root) or '.'
        remote_dir = os.path.basename(normalized_root)
        return (f'rm -rf {shlex.quote(dataset_root)}\n'
                f'huggingface-cli download {shlex.quote(cls.HF_REPO_ID)} \\\n'
                '  --repo-type dataset \\\n'
                f'  --revision {shlex.quote(cls.HF_REVISION)} \\\n'
                f'  --include {shlex.quote(remote_dir + "/*")} \\\n'
                f'  --local-dir {shlex.quote(local_dir)}')

    def _verify_dataset_version(self, dataset_root: str,
                                expected_dataset_version: str) -> None:
        version_path = os.path.join(dataset_root, self.VERSION_FILE)
        refresh_command = self._dataset_refresh_command(dataset_root)

        if not os.path.exists(version_path):
            raise RuntimeError(
                f'Dataset version file not found at {version_path}. '
                f'Expected FluxVLA dataset version '
                f'{expected_dataset_version}.\n\n'
                f'Please refresh the dataset with:\n\n{refresh_command}')

        dataset_version = self._read_dataset_version(version_path)
        if dataset_version != expected_dataset_version:
            raise RuntimeError(
                f'Dataset version mismatch for {dataset_root}. '
                f'Expected FluxVLA dataset version '
                f'{expected_dataset_version}, but found '
                f'{dataset_version or "missing"} in {version_path}.\n\n'
                f'Please refresh the dataset with:\n\n{refresh_command}')

    def _build_sample_indices(
            self, episode_fraction: float,
            selected_episodes: Optional[List[int]] = None,
            episode_frame_ranges: Optional[Dict[int, List[int]]] = None
    ) -> np.ndarray:
        if (selected_episodes is None and episode_frame_ranges is None
                and episode_fraction == 1.0):
            return np.arange(self.full_length, dtype=np.int64)

        ranges = {}
        raw_ranges = episode_frame_ranges or {}
        range_items = (raw_ranges.items() if isinstance(raw_ranges, dict)
                       else raw_ranges)
        for episode, bounds in range_items:
            if (len(bounds) != 2 or int(bounds[0]) < 0
                    or int(bounds[1]) <= int(bounds[0])):
                raise ValueError(
                    'episode_frame_ranges values must be [start, end] with '
                    f'0 <= start < end; got {episode}: {bounds}')
            ranges[int(episode)] = (int(bounds[0]), int(bounds[1]))

        episode_indices = list(self.dataset['episode_index'])
        requested = (set(int(value) for value in selected_episodes)
                     if selected_episodes is not None else None)
        sample_indices = []
        available = set()
        for start, end in zip(self.dataset_cumulative_sizes[:-1],
                              self.dataset_cumulative_sizes[1:]):
            start, end = int(start), int(end)
            local_episode_indices = episode_indices[start:end]
            ordered_episodes = list(dict.fromkeys(local_episode_indices))
            available.update(ordered_episodes)
            if requested is not None:
                keep_episodes = requested
            else:
                keep_count = int(len(ordered_episodes) * episode_fraction)
                keep_count = max(1, min(keep_count, len(ordered_episodes)))
                keep_episodes = set(ordered_episodes[:keep_count])
            for offset, episode in enumerate(local_episode_indices):
                if episode not in keep_episodes:
                    continue
                global_index = start + offset
                if episode in ranges:
                    frame_index = int(self.dataset[global_index]['frame_index'])
                    range_start, range_end = ranges[episode]
                    if not range_start <= frame_index < range_end:
                        continue
                sample_indices.append(global_index)

        if not sample_indices:
            raise ValueError('No samples left after applying episode split.')
        if requested is not None:
            missing = sorted(requested - available)
            if missing:
                raise ValueError(
                    f'Requested episode indices are absent: {missing}')
        missing_ranges = sorted(set(ranges) - available)
        if missing_ranges:
            raise ValueError(
                f'Frame ranges reference absent episodes: {missing_ranges}')
        return np.asarray(sample_indices, dtype=np.int64)

    def _resolve_index(self, index: int) -> int:
        sample_index = index % len(self.sample_indices)
        return int(self.sample_indices[sample_index])

    def _rand_another(self):
        """Randomly select another index from the dataset."""
        return int(self.sample_indices[np.random.randint(
            0, len(self.sample_indices))])

    def _get_dataset_index(self, index: int) -> int:
        """Get which dataset in data_root list the index belongs to.

        Args:
            index (int): The index in the concatenated dataset.

        Returns:
            int: The index of the dataset in data_root list (0-based).
        """
        if self.dataset_cumulative_sizes is None:
            return 0
        # Use binary search to find the index of the dataset in data_root list
        dataset_idx = np.searchsorted(
            self.dataset_cumulative_sizes, index, side='right') - 1
        return dataset_idx

    def _get_task_name(self, dataset_idx: int, index: int) -> str:
        task_idx = self.dataset[index]['task_index']
        if task_idx < 0 or task_idx >= len(self.tasks[dataset_idx]):
            return 'empty'
        return self.tasks[dataset_idx][task_idx].get('task', 'empty')

    def _invalid_start_index(self, index: int, dataset_idx: int,
                             data: Dict[str, Any]) -> bool:
        if self._get_task_name(dataset_idx, index) in ('empty', 'static'):
            return True

        first_action_index = index + self.window_start_idx
        if first_action_index == index:
            return False
        if not self._same_episode_and_dataset(first_action_index, dataset_idx,
                                              data):
            return True
        return self._get_task_name(dataset_idx,
                                   first_action_index) in ('empty', 'static')

    def _same_episode_and_dataset(self, index: int, dataset_idx: int,
                                  data: Dict[str, Any]) -> bool:
        return (index < len(self.dataset) and data['episode_index']
                == self.dataset[index]['episode_index']
                and self._get_dataset_index(index) == dataset_idx)

    def __getitem__(self, index, dataset_statistics):
        index = self._resolve_index(index)
        data = self.dataset[index]
        # Determine which dataset the data belongs to
        dataset_idx = self._get_dataset_index(index)
        while self._invalid_start_index(index, dataset_idx, data):
            index = self._rand_another()
            data = self.dataset[index]
            # Recalculate dataset_idx
            dataset_idx = self._get_dataset_index(index)
        actions = list()
        action_masks = list()
        window_idx = self.window_start_idx
        while len(actions) < self.action_window_size:
            action_index = index + window_idx
            valid_window_index = self._same_episode_and_dataset(
                action_index, dataset_idx, data)
            action_task = (
                self._get_task_name(dataset_idx, action_index)
                if valid_window_index else None)
            if valid_window_index and action_task not in ('empty', 'static'):
                if self.use_delta:
                    actions.append((
                        np.array(self.dataset[action_index][self.action_key]) -
                        np.array(self.dataset[action_index -
                                              1][self.action_key])).tolist())
                else:
                    actions.append(self.dataset[action_index][self.action_key])
                action_masks.append(1)
            elif action_task == 'empty':
                for _ in range(self.action_window_size - len(actions)):
                    actions.append(actions[-1])
                    action_masks.append(0)
                break
            elif action_task == 'static':
                window_idx += 1
                continue
            else:
                if len(actions) > 0:
                    actions.append(actions[-1])
                else:
                    actions.append(data[self.action_key])
                action_masks.append(0)
            window_idx += 1
        # Collect forward-looking frame timestamps for video models
        if self.frame_window_size > 1:
            frame_timestamps = [data['timestamp']]
            frame_masks = [1]
            for fi in range(1, self.frame_window_size):
                future_idx = index + fi * self.frame_sample_stride
                if (future_idx < len(self.dataset)
                        and self.dataset[future_idx]['episode_index']
                        == data['episode_index'] and
                        self._get_dataset_index(future_idx) == dataset_idx):
                    frame_timestamps.append(
                        self.dataset[future_idx]['timestamp'])
                    frame_masks.append(1)
                else:
                    frame_timestamps.append(frame_timestamps[-1])
                    frame_masks.append(0)
            data['frame_timestamps'] = frame_timestamps
            data['frame_masks'] = np.array(frame_masks, dtype=np.float32)

        data['info'] = self.info[dataset_idx]
        data['stats'] = dataset_statistics[self.statistic_name]
        data['actions'] = np.array(actions, dtype=np.float32)
        data['action_masks'] = np.array(action_masks, dtype=np.float32)
        if self.expose_index:
            data['index'] = np.array(index, dtype=np.int64)
        data['task_description'] = self._get_task_name(dataset_idx, index)
        data['data_root'] = self.data_root_path[dataset_idx]
        for transform in self.transforms:
            data = transform(data)

        return data

    def __len__(self):
        return self.effective_length

        # Additional initialization can be added here if needed.




