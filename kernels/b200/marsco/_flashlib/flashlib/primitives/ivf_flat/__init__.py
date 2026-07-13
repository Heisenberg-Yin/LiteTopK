"""IVF-Flat primitive -- GPU approximate nearest neighbours (inverted file).

Public API
----------
    flash_ivf_flat_build(X, nlist, *, nprobe=, niter=, ...)  -> IvfFlatIndex
    flash_ivf_flat_search(index, Q, k, *, nprobe=)           -> (vals, ids)
    flash_ivf_flat(X, Q, k, *, nlist=, nprobe=)              -> (vals, ids)

The build reuses ``flash_kmeans`` (coarse quantizer) + the x²-free
nearest-centroid assign kernel; coarse search reuses ``flash_knn``. The
new kernels are the fused ragged inverted-list fine-scans -- an
elementwise online path (:mod:`...ivf_flat.triton.fine_scan`) and a
group-by-list tensor-core path for batch throughput
(:mod:`...ivf_flat.triton.fine_scan_gemm`). Both keep the top-K on-chip
and never materialise an ``(nq x candidates)`` distance matrix, so at a
fixed ``(nlist, nprobe)`` the result matches a reference IVF-Flat
(iso-recall).

Torch fallback (CPU-OK, also the correctness oracle):
    flashlib.primitives.ivf_flat.torch_fallback.{ivf_flat_build_torch,
    ivf_flat_search_torch}
"""
from __future__ import annotations

from flashlib.primitives.ivf_flat import cost
from flashlib.primitives.ivf_flat.index import IvfFlatIndex
from flashlib.primitives.ivf_flat.impl import (
    flash_ivf_flat,
    flash_ivf_flat_build,
    flash_ivf_flat_search,
    route_op_name,
)

__all__ = [
    "IvfFlatIndex",
    "flash_ivf_flat",
    "flash_ivf_flat_build",
    "flash_ivf_flat_search",
    "route_op_name",
    "cost",
]
