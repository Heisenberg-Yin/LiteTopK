"""IVF-Flat index build (Triton/GPU path).

The build reuses two already-tuned flashlib primitives:

* coarse quantizer training -> :func:`flashlib.primitives.kmeans.flash_kmeans`
  on a sample of the database (FAISS-style ``min(M, nlist * 256)`` rows).
* full-database assignment -> :func:`...kmeans.triton.assign.euclid_assign_triton`
  (the x²-free nearest-centroid kernel; one pass over the database).

The only build-specific work is turning the per-row assignment into the
cell-contiguous CSR layout the fused fine-scan kernel consumes:

    counts   = bincount(assign)
    offsets  = [0, cumsum(counts)]           # (nlist + 1,)
    order    = argsort(assign, stable=True)  # stored-row -> original id
    data     = X[order]                      # cell-contiguous database

Padding: ``D < 16`` is zero-padded to ``Dp = 16`` (Triton ``tl.dot``
needs a contraction dim >= 16). Zero columns add 0 to every squared-L2
distance, so results are unaffected.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib.primitives.ivf_flat.index import IvfFlatIndex
from flashlib.primitives.ivf_flat.torch_fallback import _pad_features
from flashlib.primitives.kmeans import flash_kmeans
from flashlib.primitives.kmeans.triton.assign import euclid_assign_triton


def ivf_flat_build_triton(
    X: torch.Tensor,
    nlist: int,
    *,
    metric: str = "l2",
    nprobe: int = 8,
    niter: int = 20,
    train_size: Optional[int] = None,
    seed: int = 0,
) -> IvfFlatIndex:
    """Build an IVF-Flat index on the GPU. Returns :class:`IvfFlatIndex`."""
    if metric != "l2":
        raise NotImplementedError(
            f"ivf_flat currently supports metric='l2' only (got {metric!r})"
        )
    if not X.is_cuda or X.ndim != 2:
        raise ValueError("ivf_flat_build_triton requires a 2D CUDA tensor")

    M, D = X.shape
    nlist = int(min(nlist, M))
    if nlist < 1:
        raise ValueError("nlist must be >= 1")
    Dp = max(int(D), 16)
    Xp = _pad_features(X, Dp).contiguous()

    # ── coarse quantizer: k-means on a sample ──────────────────────────
    train_size = int(train_size or min(M, nlist * 256))
    train_size = max(min(train_size, M), nlist)
    g = torch.Generator(device=X.device).manual_seed(seed)
    sample_idx = torch.randperm(M, generator=g, device=X.device)[:train_size]
    sample = Xp.index_select(0, sample_idx)

    _, centroids, _ = flash_kmeans(
        sample, nlist, max_iters=niter, metric="euclidean",
    )
    centroids = centroids.contiguous()                       # (nlist, Dp)

    # ── assign every database row to its nearest centroid ──────────────
    assign = euclid_assign_triton(
        Xp.unsqueeze(0), centroids.unsqueeze(0),
    ).squeeze(0).to(torch.int64)                             # (M,)

    # ── CSR cell-contiguous layout ─────────────────────────────────────
    counts = torch.bincount(assign, minlength=nlist)         # (nlist,)
    offsets = torch.zeros(nlist + 1, dtype=torch.int64, device=X.device)
    offsets[1:] = counts.cumsum(0)
    order = torch.argsort(assign, stable=True)               # (M,) int64
    data_sorted = Xp.index_select(0, order).contiguous()     # (M, Dp)
    max_list_len = int(counts.max().item())                  # one sync at build

    return IvfFlatIndex(
        centroids=centroids,
        data=data_sorted,
        ids=order.to(torch.int64),
        list_offsets=offsets,
        metric=metric,
        D=int(D),
        Dp=int(Dp),
        nlist=int(nlist),
        nprobe=int(nprobe),
        max_list_len=max_list_len,
    )


__all__ = ["ivf_flat_build_triton"]
