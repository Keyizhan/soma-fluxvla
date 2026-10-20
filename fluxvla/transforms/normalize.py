# Derived from FluxVLA (Limx Dynamics); see LICENSE and NOTICE.
from typing import Dict, List, Optional
import numpy as np
import torch
from fluxvla.engines import TRANSFORMS

@TRANSFORMS.register_module()
class NormalizeStatesAndActions:
    """Normalize states and actions in the data.
    This transform normalizes the state and action
    dimensions in the data to match the specified
    action dimension. It pads the state and action
    dimensions to the specified action dimension.

    Args:
        action_dim (int): The dimension to which the state
            and action should be normalized.
        pad_value (float): The value to use for padding.
            Defaults to 0.0.
        norm_type (str): Type of normalization to use.
            Options: 'mean_std', 'quantile', 'min_max', or 'none'.
            Defaults to 'mean_std'.
        state_norm_type (str): Optional normalization type for states.
            Defaults to `norm_type`.
        action_norm_type (str): Optional normalization type for actions.
            Defaults to `norm_type`.
        clip_norm (bool): Whether to clip min_max/quantile normalized values
            to [-1, 1]. Defaults to False.
        normalize_states (bool): Whether to normalize states before optional
            padding/truncation. Defaults to True.
        state_key (str | None): The key in the data dictionary
            that contains the state information.
        action_key (str | None): The key in the data dictionary
            that contains the action information. If None, actions are skipped.
    """

    def __init__(self, state_key: Optional[str], action_key: Optional[str], action_dim: int=None, state_dim: int=None, norm_type: str='mean_std', state_norm_type: str=None, action_norm_type: str=None, pad_value: float=0.0, action_norm_mask: List[bool]=None, clip_norm: bool=False, normalization_epsilon: float=1e-06, normalize_states: bool=True, *args, **kwargs):
        self.state_key = state_key
        self.action_key = action_key
        self.norm_type = norm_type
        self.state_norm_type = state_norm_type or norm_type
        self.action_norm_type = action_norm_type or norm_type
        self.pad_value = pad_value
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.clip_norm = clip_norm
        self.normalization_epsilon = float(normalization_epsilon)
        self.normalize_states = normalize_states
        if action_norm_mask is not None:
            assert len(action_norm_mask) == action_dim, f'Action norm mask must be of length {action_dim}'
            self.action_norm_mask = action_norm_mask
        else:
            self.action_norm_mask = None

    def __call__(self, data: Dict) -> Dict:
        states = np.asarray(data['states'], dtype=np.float32)
        actions = None
        if self.action_key is not None and 'actions' in data:
            actions = np.asarray(data['actions'], dtype=np.float32)
        needs_state_stats = self.normalize_states and self.state_norm_type != 'none'
        needs_action_stats = actions is not None and self.action_norm_type != 'none'
        if needs_state_stats or needs_action_stats:
            assert 'stats' in data, "Input data must contain 'stats' key"
        if needs_state_stats:
            state_stats = data['stats'][self.state_key]
            states = self._normalize_by_type(states, state_stats, self.state_norm_type)
        data['states'] = states
        if actions is not None:
            action_stats = None
            if needs_action_stats:
                action_stats = data['stats'][self.action_key]
            actions = self._normalize_by_type(actions, action_stats, self.action_norm_type, self.action_norm_mask)
            data['actions'] = actions
        if self.state_dim is not None:
            data['states'] = self._pad_or_truncate_last_dim(states, self.state_dim)
        if self.action_dim is not None and actions is not None:
            data['actions'] = self._pad_or_truncate_last_dim(actions, self.action_dim)
        return data

    def _pad_or_truncate_last_dim(self, values: np.ndarray, target_dim: int) -> np.ndarray:
        current_dim = values.shape[-1]
        if current_dim >= target_dim:
            return values[..., :target_dim]
        padded_shape = (*values.shape[:-1], target_dim)
        padded = np.full(padded_shape, self.pad_value, dtype=values.dtype)
        padded[..., :current_dim] = values
        return padded

    def _normalize_by_type(self, x, stats: Dict, norm_type: str, norm_mask: List[bool]=None):
        if norm_type == 'none':
            return x
        if norm_type == 'quantile':
            return self._normalize_quantile(x, stats, norm_mask)
        if norm_type == 'min_max':
            return self._normalize_min_max(x, stats, norm_mask)
        return self._normalize(x, stats, norm_mask)

    def _normalize(self, x, stats: Dict, norm_mask: List[bool]=None):
        if norm_mask is None:
            norm_mask = [True] * x.shape[-1]
        return np.where(norm_mask, (x - np.array(stats['mean'])) / (np.array(stats['std']) + self.normalization_epsilon), x)

    def _normalize_quantile(self, x, stats: torch.tensor, norm_mask: List[bool]=None):
        assert stats['q01'] is not None
        assert stats['q99'] is not None
        if norm_mask is None:
            norm_mask = [True] * x.shape[-1]
        normalized = (x - np.array(stats['q01'])) / (np.array(stats['q99']) - np.array(stats['q01']) + self.normalization_epsilon) * 2.0 - 1.0
        if self.clip_norm:
            normalized = np.clip(normalized, -1, 1)
        return np.where(norm_mask, normalized, x)

    def _normalize_min_max(self, x, stats: Dict, norm_mask: List[bool]=None):
        assert 'min' in stats and stats['min'] is not None
        assert 'max' in stats and stats['max'] is not None
        if norm_mask is None:
            norm_mask = [True] * x.shape[-1]
        normalized = (x - np.array(stats['min'])) / (np.array(stats['max']) - np.array(stats['min']) + self.normalization_epsilon) * 2.0 - 1.0
        if self.clip_norm:
            normalized = np.clip(normalized, -1, 1)
        return np.where(norm_mask, normalized, x)
