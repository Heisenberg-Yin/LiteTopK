"""NearestNeighbors — sklearn-style wrapper around primitives.knn.

flash-knn ships a function-only API; this class adds the sklearn-compat
fit / kneighbors / kneighbors_graph interface for users who want it.
"""
from __future__ import annotations

import torch

from flashlib.primitives.knn import flash_knn_dispatch


class NearestNeighbors:
    """Nearest neighbor search.

    Args:
        n_neighbors: default k for kneighbors().
        backend: ``"triton"`` | ``"cutedsl"`` | ``"torch"`` (default: auto via
                 the smart dispatcher).

    After ``fit(X)`` the database is cached. ``kneighbors(Q)`` returns the
    ``n_neighbors`` closest rows of X for each row of Q, by squared L2.
    """

    def __init__(self, n_neighbors: int = 5, *,
                 backend: str | None = None):
        self.n_neighbors = int(n_neighbors)
        self.backend = backend
        self._X = None

    def fit(self, X: torch.Tensor) -> "NearestNeighbors":
        if not X.is_cuda or X.ndim != 2:
            raise ValueError("NearestNeighbors requires a 2D CUDA tensor")
        self._X = X
        return self

    def kneighbors(self, Q: torch.Tensor | None = None, n_neighbors: int | None = None,
                   return_distance: bool = True, *,
                   backend: str | None = None):
        if self._X is None:
            raise RuntimeError("NearestNeighbors not fitted; call fit() first.")
        k = n_neighbors if n_neighbors is not None else self.n_neighbors
        if Q is None:
            Q = self._X
        x = Q.unsqueeze(0)
        c = self._X.unsqueeze(0)
        if return_distance:
            dists, idxs = flash_knn_dispatch(
                x, c, k,
                backend=backend or self.backend,
            )
            return dists.squeeze(0), idxs.squeeze(0)
        idxs = flash_knn_dispatch(
            x, c, k,
            backend=backend or self.backend,
            return_distances=False,
        )
        return idxs.squeeze(0)


__all__ = ["NearestNeighbors"]
