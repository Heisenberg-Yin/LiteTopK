"""Ridge — sklearn-style wrapper around primitives.ridge."""
from __future__ import annotations

import torch

from flashlib.primitives.ridge import flash_ridge_regression


class Ridge:
    """Ridge regression: minimize ||y - Xw||^2 + alpha * ||w||^2.

    Args:
        alpha: regularization strength.

    Attributes set after fit():
        coef_: (D,) regression coefficients
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coef_ = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "Ridge":
        if not X.is_cuda or X.ndim != 2 or y.ndim != 1:
            raise ValueError("Ridge requires 2D X and 1D y on CUDA")
        self.coef_ = flash_ridge_regression(X, y, alpha=self.alpha)
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.coef_ is None:
            raise RuntimeError("Ridge not fitted; call fit() first.")
        return X @ self.coef_


__all__ = ["Ridge"]
