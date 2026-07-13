"""StandardScaler primitive — Triton-backed (X - mean) / std.

Public API:
    flash_standard_scaler(X)                          — alias of fit_transform
    flash_standard_scaler_fit(X)                      -> (mean, std, inv_std)
    flash_standard_scaler_transform(X, mean, inv_std) -> Y
    flash_standard_scaler_fit_transform(X)            -> (Y, (mean, std))
"""
from flashlib._lazy import lazy_attr
from flashlib.primitives.standard_scaler import cost
from flashlib.primitives.standard_scaler.impl import (
    flash_standard_scaler,
    flash_standard_scaler_fit,
    flash_standard_scaler_transform,
    flash_standard_scaler_fit_transform,
)


cutedsl_standard_scaler_fit = lazy_attr(
    "flashlib.primitives.standard_scaler.cutedsl", "cutedsl_standard_scaler_fit",
)
cutedsl_standard_scaler_transform = lazy_attr(
    "flashlib.primitives.standard_scaler.cutedsl", "cutedsl_standard_scaler_transform",
)
cutedsl_standard_scaler_fit_transform = lazy_attr(
    "flashlib.primitives.standard_scaler.cutedsl", "cutedsl_standard_scaler_fit_transform",
)
cutedsl_available = lazy_attr(
    "flashlib.primitives.standard_scaler.cutedsl", "cutedsl_available",
)


__all__ = [
    "flash_standard_scaler",
    "flash_standard_scaler_fit",
    "flash_standard_scaler_transform",
    "flash_standard_scaler_fit_transform",
    "cutedsl_standard_scaler_fit",
    "cutedsl_standard_scaler_transform",
    "cutedsl_standard_scaler_fit_transform",
    "cutedsl_available",
    "cost",
]
