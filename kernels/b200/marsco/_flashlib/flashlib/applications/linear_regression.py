"""LinearRegression — sklearn-style wrapper around primitives.linear_regression."""
from __future__ import annotations

import torch

from flashlib.primitives.linear_regression import flash_linear_regression


class LinearRegression:
    """Ordinary least squares regression via normal equations.

    Attributes set after fit():
        coef_:        (D,) regression coefficients
        intercept_:   scalar (always 0 in this version; pass an X with a bias column)
    """

    def __init__(self):
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "LinearRegression":
        if not X.is_cuda or X.ndim != 2 or y.ndim != 1:
            raise ValueError("LinearRegression requires 2D X and 1D y on CUDA")
        self.coef_ = flash_linear_regression(X, y)
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.coef_ is None:
            raise RuntimeError("LinearRegression not fitted; call fit() first.")
        return X @ self.coef_ + self.intercept_


__all__ = ["LinearRegression"]
