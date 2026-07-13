"""HDBSCAN primitive -- pairwise MRD + Boruvka MST + tree condensation.

Public API:
    flash_hdbscan(X, min_cluster_size=25, min_samples=5, *,
                  approximate=True, prefer="auto", k=32, tol=1e-3,
                  backend=None)
        -> (N,) int32 cluster labels (-1 for noise).
"""
from flashlib._lazy import lazy_attr
from flashlib.primitives.hdbscan import cost
from flashlib.primitives.hdbscan.impl import (
    flash_hdbscan,
    flash_hdbscan_sparse,
    triton_hdbscan_mrd,
)


cutedsl_hdbscan = lazy_attr(
    "flashlib.primitives.hdbscan.cutedsl",
    "cutedsl_hdbscan",
)
cutedsl_fused_mrd_edges = lazy_attr(
    "flashlib.primitives.hdbscan.cutedsl",
    "cutedsl_fused_mrd_edges",
)


__all__ = [
    "flash_hdbscan",
    "flash_hdbscan_sparse",
    "triton_hdbscan_mrd",
    "cutedsl_hdbscan",
    "cutedsl_fused_mrd_edges",
    "cost",
]
