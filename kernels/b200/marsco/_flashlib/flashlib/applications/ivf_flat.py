"""IVFFlat -- sklearn-style wrapper around primitives.ivf_flat.

The functional API (``flash_ivf_flat_build`` / ``flash_ivf_flat_search``)
is the fast path; this class adds the familiar ``fit`` / ``kneighbors``
interface, caching the built index between queries.

Example
-------
    from flashlib.applications import IVFFlat
    index = IVFFlat(nlist=1024, nprobe=16).fit(database)   # (M, D) CUDA
    dist, idx = index.kneighbors(queries, n_neighbors=10)  # squared L2
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib.primitives.ivf_flat import (
    flash_ivf_flat_build,
    flash_ivf_flat_search,
)


class IVFFlat:
    """Approximate nearest neighbours via an inverted-file (IVF) index.

    Args:
        nlist: number of inverted lists / coarse centroids (clamped to ``M``).
        nprobe: lists probed per query at search time (recall knob).
        n_neighbors: default ``k`` for :meth:`kneighbors`.
        metric: distance metric (``"l2"`` only in v1).
        niter: Lloyd iterations for the coarse quantizer.
        train_size: rows sampled to train the quantizer
            (default ``min(M, nlist*256)``).
        seed: RNG seed (deterministic build).
        backend: ``"triton"`` | ``"torch"`` (default: auto).

    Recall is fixed by ``(nlist, nprobe)``; at a fixed pair the result
    matches a reference IVF-Flat, so tuning ``nprobe`` trades recall for
    speed without changing the kernel.
    """

    def __init__(
        self,
        nlist: int = 1024,
        *,
        nprobe: int = 8,
        n_neighbors: int = 5,
        metric: str = "l2",
        niter: int = 20,
        train_size: Optional[int] = None,
        seed: int = 0,
        backend: Optional[str] = None,
    ):
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.niter = int(niter)
        self.train_size = train_size
        self.seed = int(seed)
        self.backend = backend
        self.index_ = None

    def fit(self, X: torch.Tensor) -> "IVFFlat":
        """Build the index over database ``X`` of shape ``(M, D)``."""
        if X.ndim != 2:
            raise ValueError("IVFFlat requires a 2D (M, D) tensor")
        self.index_ = flash_ivf_flat_build(
            X, self.nlist, metric=self.metric, nprobe=self.nprobe,
            niter=self.niter, train_size=self.train_size, seed=self.seed,
            backend=self.backend,
        )
        return self

    def kneighbors(
        self,
        Q: torch.Tensor,
        n_neighbors: Optional[int] = None,
        return_distance: bool = True,
        *,
        nprobe: Optional[int] = None,
        variant: str = "auto",
    ):
        """Return the ``k`` nearest neighbours of each row of ``Q``.

        Returns ``(distances, indices)`` (squared L2) when
        ``return_distance`` else just ``indices``. ``variant`` forces the
        fine-scan kernel (``"auto"`` | ``"gemm"`` | ``"elementwise"``).
        """
        if self.index_ is None:
            raise RuntimeError("IVFFlat not fitted; call fit() first.")
        k = n_neighbors if n_neighbors is not None else self.n_neighbors
        vals, ids = flash_ivf_flat_search(
            self.index_, Q, k, nprobe=nprobe or self.nprobe,
            backend=self.backend, variant=variant,
        )
        if return_distance:
            return vals, ids
        return ids

    # sklearn-NearestNeighbors alias.
    def fit_kneighbors(self, X, Q, n_neighbors=None, **kw):
        return self.fit(X).kneighbors(Q, n_neighbors=n_neighbors, **kw)


__all__ = ["IVFFlat"]
