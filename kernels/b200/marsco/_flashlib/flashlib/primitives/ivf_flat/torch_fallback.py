"""Pure-torch IVF-Flat reference (CPU-OK, deterministic).

Used as

* the ``backend="torch"`` fallback for :func:`flash_ivf_flat_build` /
  :func:`flash_ivf_flat_search` when CUDA is unavailable, and
* the correctness oracle in the test-suite: given the **same**
  :class:`~flashlib.primitives.ivf_flat.index.IvfFlatIndex`,
  :func:`ivf_flat_search_torch` performs the identical coarse + fine
  computation as the Triton kernel, so any divergence is a kernel bug
  rather than an algorithm difference.

No Triton import; uses a small Lloyd k-means so it works without a GPU.
"""
from __future__ import annotations

from typing import Optional

import torch


def _pad_features(x: torch.Tensor, Dp: int) -> torch.Tensor:
    """Zero-pad the trailing feature dim to ``Dp`` (no-op when already wide)."""
    D = x.shape[-1]
    if D >= Dp:
        return x
    pad = torch.zeros((*x.shape[:-1], Dp - D), device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)


def _lloyd_kmeans(
    sample: torch.Tensor, nlist: int, *, niter: int, seed: int
) -> torch.Tensor:
    """Tiny Lloyd k-means returning ``(nlist, D)`` centroids (fp32 math)."""
    n = sample.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=g)[:nlist]
    centroids = sample[perm.to(sample.device)].to(torch.float32).clone()
    s = sample.to(torch.float32)
    for _ in range(max(1, niter)):
        d2 = torch.cdist(s, centroids) ** 2          # (n, nlist)
        assign = d2.argmin(dim=1)                    # (n,)
        new = centroids.clone()
        for c in range(nlist):
            mask = assign == c
            if bool(mask.any()):
                new[c] = s[mask].mean(dim=0)
        shift = (new - centroids).norm(dim=-1).max()
        centroids = new
        if float(shift) == 0.0:
            break
    return centroids.to(sample.dtype)


def ivf_flat_build_torch(
    X: torch.Tensor,
    nlist: int,
    *,
    metric: str = "l2",
    nprobe: int = 8,
    niter: int = 20,
    train_size: Optional[int] = None,
    seed: int = 0,
):
    """Build an :class:`IvfFlatIndex` with pure torch ops."""
    from flashlib.primitives.ivf_flat.index import IvfFlatIndex

    if metric != "l2":
        raise NotImplementedError(f"ivf_flat torch supports metric='l2' only (got {metric!r})")
    if X.ndim != 2:
        raise ValueError("ivf_flat build expects a 2D (M, D) tensor")
    M, D = X.shape
    nlist = int(min(nlist, M))
    Dp = max(int(D), 16)
    Xp = _pad_features(X, Dp).contiguous()

    train_size = int(train_size or min(M, nlist * 256))
    train_size = max(train_size, nlist)
    g = torch.Generator(device="cpu").manual_seed(seed)
    sample_idx = torch.randperm(M, generator=g)[:train_size].to(X.device)
    sample = Xp.index_select(0, sample_idx)

    centroids = _lloyd_kmeans(sample, nlist, niter=niter, seed=seed)  # (nlist, Dp)

    d2 = torch.cdist(Xp.to(torch.float32), centroids.to(torch.float32)) ** 2
    assign = d2.argmin(dim=1)                                  # (M,) int64

    counts = torch.bincount(assign, minlength=nlist)           # (nlist,)
    offsets = torch.zeros(nlist + 1, dtype=torch.int64, device=X.device)
    offsets[1:] = counts.cumsum(0)
    order = torch.argsort(assign, stable=True)                 # (M,) int64
    data_sorted = Xp.index_select(0, order).contiguous()

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
        max_list_len=int(counts.max().item()),
    )


def ivf_flat_search_torch(index, Q: torch.Tensor, k: int, *, nprobe: Optional[int] = None):
    """Reference coarse + fine IVF-Flat search over a built index.

    Returns ``(vals, ids)`` where ``vals[i, j]`` is the true squared-L2
    distance to the ``j``-th nearest neighbour and ``ids`` are original
    row ids. Mirrors the Triton kernel exactly: probe the ``nprobe``
    nearest lists, scan their members, keep the global top-``k``.
    """
    nprobe = int(nprobe or index.nprobe)
    if Q.ndim != 2:
        raise ValueError("ivf_flat search expects a 2D (nq, D) query tensor")
    nq = Q.shape[0]
    Dp = index.Dp
    Qp = _pad_features(Q.to(index.data.dtype), Dp).to(torch.float32)
    cents = index.centroids.to(torch.float32)
    data = index.data.to(torch.float32)
    offsets = index.list_offsets

    nprobe = min(nprobe, index.nlist)
    # Coarse: nprobe nearest centroids per query.
    coarse_d2 = torch.cdist(Qp, cents) ** 2                    # (nq, nlist)
    probed = coarse_d2.topk(nprobe, dim=1, largest=False).indices  # (nq, nprobe)

    out_vals = torch.full((nq, k), float("inf"), device=Q.device, dtype=torch.float32)
    out_ids = torch.full((nq, k), -1, device=Q.device, dtype=torch.int64)

    for i in range(nq):
        cand_pos = []
        for p in range(nprobe):
            c = int(probed[i, p].item())
            s, e = int(offsets[c].item()), int(offsets[c + 1].item())
            if e > s:
                cand_pos.append(torch.arange(s, e, device=Q.device))
        if not cand_pos:
            continue
        pos = torch.cat(cand_pos)                              # (n_cand,)
        diff = Qp[i][None, :] - data.index_select(0, pos)      # (n_cand, Dp)
        dist = (diff * diff).sum(dim=1)                        # (n_cand,)
        kk = min(k, dist.shape[0])
        vals, sel = dist.topk(kk, largest=False)
        out_vals[i, :kk] = vals
        out_ids[i, :kk] = index.ids[pos[sel]]

    return out_vals, out_ids


__all__ = ["ivf_flat_build_torch", "ivf_flat_search_torch", "_pad_features"]
