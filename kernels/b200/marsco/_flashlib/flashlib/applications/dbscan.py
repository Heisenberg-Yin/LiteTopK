"""DBSCAN — sklearn-style wrapper around primitives.dbscan."""
from __future__ import annotations

import torch

from flashlib.primitives.dbscan import flash_dbscan


class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise.

    Args:
        eps: neighborhood radius (Euclidean).
        min_samples: minimum points in eps-neighborhood for a core point.
        max_neighbors: cap on neighbors considered per point (default 32).

    Attributes set after fit():
        labels_: (N,) int32 — cluster id (>= 0) or -1 for noise.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 max_neighbors: int = 32, *,
                 backend: str | None = None, variant: str | None = None):
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.max_neighbors = int(max_neighbors)
        self.backend = backend
        self.variant = variant
        self.labels_ = None

    def fit(self, X: torch.Tensor, **kwargs) -> "DBSCAN":
        if not X.is_cuda or X.dtype != torch.float32:
            raise ValueError("DBSCAN requires a 2D CUDA float32 tensor")
        kwargs.setdefault("backend", self.backend)
        kwargs.setdefault("variant", self.variant)
        # flash_dbscan only accepts known kwargs — drop any None passthroughs.
        passthrough = {k: v for k, v in kwargs.items() if v is not None}
        self.labels_ = flash_dbscan(
            X, eps=self.eps, min_samples=self.min_samples,
            max_neighbors=self.max_neighbors, **passthrough,
        )
        return self

    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.labels_


__all__ = ["DBSCAN"]
