"""SpectralClustering — sklearn-style wrapper around primitives.spectral_clustering."""
from __future__ import annotations

import torch

from flashlib.primitives.spectral_clustering import flash_spectral_clustering


class SpectralClustering:
    """Spectral clustering: KNN graph + Laplacian eigendecomposition + KMeans.

    Args:
        n_clusters: number of clusters.
        n_neighbors: KNN graph size.

    Attributes set after fit():
        labels_: (N,) int — cluster ids in [0, n_clusters).
    """

    def __init__(self, n_clusters: int = 8, n_neighbors: int = 10):
        self.n_clusters = int(n_clusters)
        self.n_neighbors = int(n_neighbors)
        self.labels_ = None

    def fit(self, X: torch.Tensor) -> "SpectralClustering":
        if not X.is_cuda or X.ndim != 2:
            raise ValueError("SpectralClustering requires a 2D CUDA tensor")
        self.labels_ = flash_spectral_clustering(X, n_clusters=self.n_clusters,
                                                 n_neighbors=self.n_neighbors)
        return self

    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.labels_


__all__ = ["SpectralClustering"]
