"""StandardScaler dispatcher.

Triton is the only auto-routed backend (always available, fast). The
CuteDSL backend is opt-in via ``backend="cutedsl"`` and falls back to
Triton when the CUTLASS DSL is unavailable on the running install.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib.primitives.standard_scaler.cutedsl import (
    cutedsl_available,
    cutedsl_standard_scaler_fit,
    cutedsl_standard_scaler_fit_transform,
    cutedsl_standard_scaler_transform,
)
from flashlib.primitives.standard_scaler.triton import (
    flash_standard_scaler_fit as triton_standard_scaler_fit,
    flash_standard_scaler_transform as triton_standard_scaler_transform,
    flash_standard_scaler_fit_transform as triton_standard_scaler_fit_transform,
)


def _use_cutedsl(backend: Optional[str]) -> bool:
    return backend == "cutedsl" and bool(cutedsl_available())


def flash_standard_scaler_fit(
    X: torch.Tensor,
    *,
    backend: Optional[str] = None,
    fused: bool = True,
):
    if _use_cutedsl(backend):
        return cutedsl_standard_scaler_fit(X)
    return triton_standard_scaler_fit(X, fused=fused)


def flash_standard_scaler_transform(
    X: torch.Tensor,
    mean: torch.Tensor,
    inv_std: torch.Tensor,
    *,
    backend: Optional[str] = None,
):
    if _use_cutedsl(backend):
        return cutedsl_standard_scaler_transform(X, mean, inv_std)
    return triton_standard_scaler_transform(X, mean, inv_std)


def flash_standard_scaler_fit_transform(
    X: torch.Tensor,
    *,
    backend: Optional[str] = None,
):
    if _use_cutedsl(backend):
        return cutedsl_standard_scaler_fit_transform(X)
    return triton_standard_scaler_fit_transform(X)


flash_standard_scaler = flash_standard_scaler_fit_transform
