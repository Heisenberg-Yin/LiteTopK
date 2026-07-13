"""TSNE — sklearn-style wrapper around primitives.tsne."""
from __future__ import annotations

import torch

from flashlib.primitives.tsne import flash_tsne


class TSNE:
    """t-Distributed Stochastic Neighbor Embedding.

    Args:
        n_components: output dimensionality (2 or 3).
        perplexity: kernel bandwidth for P matrix construction.
        n_iter: SGD iterations.
        learning_rate: SGD learning rate.

    Attributes set after fit():
        embedding_: (N, n_components) low-dim embedding.
    """

    def __init__(self, n_components: int = 2, perplexity: float = 30.0,
                 n_iter: int = 1000, learning_rate: float = 200.0):
        self.n_components = int(n_components)
        self.perplexity = float(perplexity)
        self.n_iter = int(n_iter)
        self.learning_rate = float(learning_rate)
        self.embedding_ = None

    def fit(self, X: torch.Tensor) -> "TSNE":
        if not X.is_cuda or X.ndim != 2:
            raise ValueError("TSNE requires a 2D CUDA tensor")
        self.embedding_ = flash_tsne(X, n_iter=self.n_iter, lr=self.learning_rate,
                                     perplexity=self.perplexity)
        return self

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.embedding_


__all__ = ["TSNE"]
