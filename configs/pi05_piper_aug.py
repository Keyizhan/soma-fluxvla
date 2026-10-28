"""Ordinary PI0.5 training with online colour jitter and no spatial crop.

Each execution creates an independent configuration. Subtask recipes are
not modified. Train and serve both use the full frame.
"""
_base_ = []  # Explicit eager mode for MMEngine; this is a values-only config.
import runpy as _runpy

globals().update({k: v for k, v in _runpy.run_module(
    'fluxvla.training.presets.pi05_piper_aug').items() if not k.startswith('_')})
