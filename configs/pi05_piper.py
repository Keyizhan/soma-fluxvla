"""Compatibility config; each execution creates an independent configuration."""
_base_ = []  # Explicit eager mode for MMEngine; this is a values-only config.
import runpy as _runpy

globals().update({k: v for k, v in _runpy.run_module(
    "fluxvla.training.presets.pi05_piper").items() if not k.startswith("_")})
