"""MultinomialNB — sklearn-style wrapper around primitives.multinomial_nb."""
from __future__ import annotations

import torch

from flashlib.primitives.multinomial_nb import (
    flash_multinomial_nb_fit,
    flash_multinomial_nb_predict,
)


class MultinomialNB:
    """Multinomial Naive Bayes for discrete features (e.g. token counts).

    Args:
        alpha: Laplace smoothing strength.

    Attributes set after fit():
        params_: dict produced by flash_multinomial_nb_fit (class log priors,
                 feature log probabilities).
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.params_ = None

    def fit(self, X: torch.Tensor, y: torch.Tensor, n_classes: int | None = None) -> "MultinomialNB":
        if not X.is_cuda or X.ndim != 2 or y.ndim != 1:
            raise ValueError("MultinomialNB requires 2D X and 1D y on CUDA")
        K = int(n_classes) if n_classes is not None else int(y.max().item()) + 1
        self.params_ = flash_multinomial_nb_fit(X, y, n_classes=K, alpha=self.alpha)
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.params_ is None:
            raise RuntimeError("MultinomialNB not fitted; call fit() first.")
        return flash_multinomial_nb_predict(X, self.params_)


__all__ = ["MultinomialNB"]
