"""DBSCAN primitive — flash_knn radius filter + connected components.

Public API:
    flash_dbscan(X, eps, min_samples=5, max_neighbors=32, *, tol=1e-3, backend=None)
        -> (N,) int32 component labels. Default routes to the Triton
        backend with ``flash_cc_from_edges`` (BLOCK=128, 4 warps).
    cutedsl_dbscan                        — CuteDSL grid-radius variant
    cutedsl_grid_radius_search            — standalone grid kernel
"""
from flashlib._lazy import lazy_attr
from flashlib.primitives.dbscan import cost
from flashlib.primitives.dbscan.impl import flash_dbscan


cutedsl_dbscan = lazy_attr(
    "flashlib.primitives.dbscan.cutedsl", "cutedsl_dbscan",
)
cutedsl_grid_radius_search = lazy_attr(
    "flashlib.primitives.dbscan.cutedsl", "cutedsl_grid_radius_search",
)


__all__ = [
    "flash_dbscan",
    "cutedsl_dbscan",
    "cutedsl_grid_radius_search",
    "cost",
]
