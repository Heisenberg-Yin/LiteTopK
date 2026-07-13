"""StandardScaler — sklearn-style wrapper around primitives.standard_scaler."""
from __future__ import annotations

import torch

from flashlib.primitives.standard_scaler import (
    flash_standard_scaler_fit,
    flash_standard_scaler_transform,
)


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance.

    Matches sklearn.preprocessing.StandardScaler semantics: biased std (ddof=0),
    zero-std columns are passed through unchanged.

    Attributes set after fit():
        mean_     : (D,) fp32  per-column mean
        scale_    : (D,) fp32  per-column std (sklearn `scale_`)
        var_      : (D,) fp32  per-column variance (sklearn `var_`)
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.var_ = None
        self._inv_std = None

    def fit(self, X: torch.Tensor) -> "StandardScaler":
        if not X.is_cuda or X.ndim != 2:
            raise ValueError("StandardScaler requires a 2D CUDA tensor")
        mean, std, inv_std = flash_standard_scaler_fit(X)
        self.mean_ = mean
        self.scale_ = std
        self.var_ = std * std
        self._inv_std = inv_std
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self._inv_std is None:
            raise RuntimeError("StandardScaler not fitted; call fit() first.")
        return flash_standard_scaler_transform(X, self.mean_, self._inv_std)

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)


__all__ = ["StandardScaler"]
