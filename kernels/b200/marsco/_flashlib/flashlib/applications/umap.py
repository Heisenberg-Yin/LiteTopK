"""UMAP — sklearn-style wrapper around primitives.umap."""
from __future__ import annotations

import torch

from flashlib.primitives.umap import flash_umap


class UMAP:
    """Uniform Manifold Approximation and Projection.

    Args:
        n_neighbors: number of neighbors for fuzzy simplicial set.
        n_components: output embedding dimensionality.
        n_epochs: SGD epochs for layout optimization.

    Attributes set after fit():
        embedding_: (N, n_components) low-dim embedding.
    """

    def __init__(self, n_neighbors: int = 15, n_components: int = 2, n_epochs: int = 200):
        self.n_neighbors = int(n_neighbors)
        self.n_components = int(n_components)
        self.n_epochs = int(n_epochs)
        self.embedding_ = None

    def fit(self, X: torch.Tensor) -> "UMAP":
        if not X.is_cuda or X.ndim != 2:
            raise ValueError("UMAP requires a 2D CUDA tensor")
        self.embedding_ = flash_umap(X, n_neighbors=self.n_neighbors,
                                     n_components=self.n_components,
                                     n_epochs=self.n_epochs)
        return self

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.embedding_


__all__ = ["UMAP"]
