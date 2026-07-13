"""IVF-Flat dispatcher + routing rule.

Public entry points:

* :func:`flash_ivf_flat_build`  -- train coarse quantizer + lay out the
  cell-contiguous inverted lists, returning an :class:`IvfFlatIndex`.
* :func:`flash_ivf_flat_search` -- coarse (``flash_knn`` over centroids)
  + fused fine-scan, returning ``(vals, ids)``.
* :func:`flash_ivf_flat`        -- one-shot ``build`` then ``search``
  convenience, mirroring the ``flash_knn(x, c, k)`` ergonomics.

IVF-Flat is an *approximate* nearest-neighbour method, but the
approximation lives entirely in the ``(nlist, nprobe)`` parameters: at a
fixed pair, the probed candidate set -- and thus recall -- is identical
to a reference IVF-Flat (FAISS / cuVS). The speedup comes from the fused,
no-materialisation fine-scan kernels (reusing flash_kmeans + flash_knn
for build and coarse search), not from changing what is computed.

Backends
--------
* ``backend="triton"`` (default on CUDA) -- the Triton build + fused
  fine-scan kernels.
* ``backend="torch"``  (default on CPU)  -- the pure-torch reference,
  also the correctness oracle.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib import _hw
from flashlib.primitives.ivf_flat.index import IvfFlatIndex
from flashlib.primitives.ivf_flat import torch_fallback
from flashlib.primitives.ivf_flat.triton.build import ivf_flat_build_triton
from flashlib.primitives.ivf_flat.triton.search import ivf_flat_search_triton


Backend = str


def _route(
    *,
    backend: Optional[str] = None,
    hw: Optional[_hw.HwProps] = None,
) -> Backend:
    """Pick a backend: Triton on CUDA, torch otherwise (override-able)."""
    if backend is not None:
        return backend
    hw = hw or _hw.current()
    return "triton" if hw.is_cuda else "torch"


_OP_NAME = {
    "triton": "ivf_flat_triton",
    "torch": "ivf_flat_torch",
}


def route_op_name(
    *, M: int, D: int, nlist: int, nprobe: int, k: int,
    hw: Optional[_hw.HwProps] = None,
) -> str:
    """Canonical op_name the runtime dispatcher would pick (for the cost API)."""
    del M, D, nlist, nprobe, k
    return _OP_NAME[_route(hw=hw)]


def flash_ivf_flat_build(
    X: torch.Tensor,
    nlist: int,
    *,
    metric: str = "l2",
    nprobe: int = 8,
    niter: int = 20,
    train_size: Optional[int] = None,
    seed: int = 0,
    backend: Optional[str] = None,
) -> IvfFlatIndex:
    """Build an IVF-Flat index from database ``X`` of shape ``(M, D)``.

    Parameters
    ----------
    X : (M, D) tensor
        Database vectors. CUDA for the Triton path; CPU routes to torch.
    nlist : int
        Number of inverted lists / coarse centroids (clamped to ``M``).
    metric : {"l2"}
        Distance metric (squared-L2 only).
    nprobe : int, default 8
        Default number of lists to probe at search time (overridable per
        :func:`flash_ivf_flat_search` call).
    niter : int, default 20
        Lloyd iterations for the coarse quantizer.
    train_size : int, optional
        Rows sampled to train the quantizer (default ``min(M, nlist*256)``).
    seed : int, default 0
        RNG seed for sampling + k-means init (deterministic build).
    backend : {"triton", "torch"}, optional
        Override the auto-route.
    """
    chosen = _route(backend=backend)
    fn = ivf_flat_build_triton if chosen == "triton" else torch_fallback.ivf_flat_build_torch
    return fn(
        X, nlist, metric=metric, nprobe=nprobe,
        niter=niter, train_size=train_size, seed=seed,
    )


def flash_ivf_flat_search(
    index: IvfFlatIndex,
    Q: torch.Tensor,
    k: int,
    *,
    nprobe: Optional[int] = None,
    backend: Optional[str] = None,
    variant: str = "auto",
):
    """Search a built ``index`` for the ``k`` nearest neighbours of ``Q``.

    Returns ``(vals, ids)`` with ``vals[i, j]`` the true squared-L2
    distance and ``ids`` the caller's original row ids (``-1`` padded
    when a query has fewer than ``k`` probed candidates).

    ``variant`` selects the fine-scan kernel on the Triton path:
    ``"auto"`` (default) picks the tensor-core group-by-list GEMM for
    batched search and the elementwise kernel for online/tiny-batch;
    ``"gemm"`` / ``"elementwise"`` force one.
    """
    chosen = _route(backend=backend)
    if chosen == "triton" and Q.is_cuda and index.data.is_cuda:
        return ivf_flat_search_triton(index, Q, k, nprobe=nprobe, variant=variant)
    return torch_fallback.ivf_flat_search_torch(index, Q, k, nprobe=nprobe)


def flash_ivf_flat(
    X: torch.Tensor,
    Q: torch.Tensor,
    k: int,
    *,
    nlist: int = 1024,
    nprobe: int = 8,
    metric: str = "l2",
    niter: int = 20,
    train_size: Optional[int] = None,
    seed: int = 0,
    backend: Optional[str] = None,
):
    """One-shot build + search convenience.

    Equivalent to :func:`flash_ivf_flat_build` followed by
    :func:`flash_ivf_flat_search`; returns ``(vals, ids)``. For repeated
    queries against the same database, build once and reuse the index.
    """
    index = flash_ivf_flat_build(
        X, nlist, metric=metric, nprobe=nprobe,
        niter=niter, train_size=train_size, seed=seed, backend=backend,
    )
    return flash_ivf_flat_search(index, Q, k, nprobe=nprobe, backend=backend)


__all__ = [
    "IvfFlatIndex",
    "flash_ivf_flat",
    "flash_ivf_flat_build",
    "flash_ivf_flat_search",
    "route_op_name",
]
