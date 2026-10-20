"""Image and state preprocessing matching the subtask training fixture."""
import json
from pathlib import Path

import numpy as np


def load_state_stats(path: str, state_dim: int = 14) -> dict:
    """Require training state quantiles, separately from action quantiles."""
    stats = json.loads(Path(path).expanduser().read_text())
    if 'private' in stats:
        stats = stats['private']
    state = stats.get('state')
    if not isinstance(state, dict):
        raise ValueError('subtask norm_stats must include training state.q01/q99')
    low = np.asarray(state['q01'], dtype=np.float32)
    high = np.asarray(state['q99'], dtype=np.float32)
    if (low.shape != (state_dim,) or high.shape != (state_dim,)
            or not np.isfinite(low).all() or not np.isfinite(high).all()
            or np.any(high < low)):
        raise ValueError('invalid subtask state quantiles')
    return {'q01': low, 'q99': high}


def normalize_state(state: np.ndarray, stats: dict) -> np.ndarray:
    """Exact fixture convention: float32 quantile scaling, no clipping."""
    state = np.asarray(state, dtype=np.float32)
    low = np.asarray(stats['q01'], dtype=np.float32)
    high = np.asarray(stats['q99'], dtype=np.float32)
    if state.shape != low.shape or not np.isfinite(state).all():
        raise ValueError('state must be finite and match the training quantiles')
    span = np.where(high - low < 1e-6, 1.0, high - low)
    return (2 * (state - low) / span - 1).astype(np.float32)


def resize_rgb(frame: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Use the training fixture's PIL bilinear resize, preserving RGB order."""
    from PIL import Image

    frame = np.asarray(frame)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('subtask image must be uint8 HWC RGB')
    height, width = size
    if frame.shape[:2] != (height, width):
        frame = np.asarray(Image.fromarray(frame).resize(
            (width, height), resample=Image.Resampling.BILINEAR))
    return np.ascontiguousarray(frame)
