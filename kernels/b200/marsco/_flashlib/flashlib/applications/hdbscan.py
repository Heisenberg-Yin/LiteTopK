"""HDBSCAN — sklearn-style wrapper around primitives.hdbscan."""
from __future__ import annotations

import torch

from flashlib.primitives.hdbscan import flash_hdbscan


class HDBSCAN:
    """Hierarchical density-based clustering.

    Args:
        min_cluster_size: minimum cluster size.
        min_samples: smoothing parameter for core distances; defaults to min_cluster_size.

    Attributes set after fit():
        labels_: (N,) int — cluster id (>= 0) or -1 for noise.
    """

    def __init__(self, min_cluster_size: int = 5, min_samples: int | None = None):
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = int(min_samples) if min_samples is not None else None
        self.labels_ = None

    def fit(self, X: torch.Tensor) -> "HDBSCAN":
        if not X.is_cuda or X.dtype != torch.float32 or X.ndim != 2:
            raise ValueError("HDBSCAN requires a 2D CUDA float32 tensor")
        ms = self.min_samples or self.min_cluster_size
        self.labels_ = flash_hdbscan(X, min_cluster_size=self.min_cluster_size,
                                     min_samples=ms)
        return self

    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.labels_


__all__ = ["HDBSCAN"]
