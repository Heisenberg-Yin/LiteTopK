"""UMAP primitive — KNN graph + fuzzy simplicial set + SGD layout optimization.

Public API:
    flash_umap(X, n_neighbors=15, ..., *, tol=1e-3, backend=None)
        -> (N, n_components) embedding. Routes to the Triton backend by
        default with bf16 storage for the KNN GEMM (selected via
        :func:`flashlib.linalg.gemm.storage_dtype_for`).
    cutedsl_flash_umap                    — opt-in CuteDSL fused-fuzzy variant
    cutedsl_smooth_knn_dist               — CuteDSL smooth-knn bisect kernel
    cutedsl_umap_fuzzy_simplicial_set     — CuteDSL fused fuzzy simplicial set
"""
from flashlib._lazy import lazy_attr
from flashlib.primitives.umap import cost
from flashlib.primitives.umap.impl import flash_umap


cutedsl_flash_umap = lazy_attr(
    "flashlib.primitives.umap.cutedsl", "cutedsl_flash_umap",
)
cutedsl_smooth_knn_dist = lazy_attr(
    "flashlib.primitives.umap.cutedsl", "cutedsl_smooth_knn_dist",
)
cutedsl_umap_fuzzy_simplicial_set = lazy_attr(
    "flashlib.primitives.umap.cutedsl", "cutedsl_umap_fuzzy_simplicial_set",
)
cutedsl_available = lazy_attr(
    "flashlib.primitives.umap.cutedsl", "cutedsl_available",
)


__all__ = [
    "flash_umap",
    "cutedsl_flash_umap",
    "cutedsl_smooth_knn_dist",
    "cutedsl_umap_fuzzy_simplicial_set",
    "cutedsl_available",
    "cost",
]
