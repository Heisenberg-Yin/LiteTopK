"""SpectralClustering primitive — KNN graph + Laplacian eigendecomp + KMeans.

Public API:
    flash_spectral_clustering(X, n_clusters, *, tol=1e-3, backend=None)
        -> (N,) int64 cluster labels. Routes to the Triton backend by
        default with bf16 KNN storage (selected via
        :func:`flashlib.linalg.gemm.storage_dtype_for`).
    triton_spectral_clustering             — embedding-only entry point
                                             (returns the (N, n_clusters)
                                             eigvec matrix without KMeans)
    cutedsl_spectral_clustering            — opt-in CuteDSL fused variant
    cutedsl_power_iter_top_k               — CuteDSL power-iter kernel
    cutedsl_power_iter_top_k_fused         — CuteDSL fused power-iter kernel
    cutedsl_qmul_eigvecs / qmul_rownorm    — CuteDSL Q multiply helpers
    cutedsl_row_l2_normalize               — CuteDSL row-normalise kernel
"""
from flashlib._lazy import lazy_attr
from flashlib.primitives.spectral_clustering import cost
from flashlib.primitives.spectral_clustering.impl import (
    flash_spectral_clustering,
    triton_spectral_clustering,
)


cutedsl_spectral_clustering = lazy_attr(
    "flashlib.primitives.spectral_clustering.cutedsl",
    "cutedsl_spectral_clustering",
)
cutedsl_power_iter_top_k = lazy_attr(
    "flashlib.primitives.spectral_clustering.cutedsl",
    "cutedsl_power_iter_top_k",
)
cutedsl_power_iter_top_k_fused = lazy_attr(
    "flashlib.primitives.spectral_clustering.cutedsl",
    "cutedsl_power_iter_top_k_fused",
)
cutedsl_qmul_eigvecs = lazy_attr(
    "flashlib.primitives.spectral_clustering.cutedsl",
    "cutedsl_qmul_eigvecs",
)
cutedsl_qmul_rownorm = lazy_attr(
    "flashlib.primitives.spectral_clustering.cutedsl",
    "cutedsl_qmul_rownorm",
)
cutedsl_row_l2_normalize = lazy_attr(
    "flashlib.primitives.spectral_clustering.cutedsl",
    "cutedsl_row_l2_normalize",
)


__all__ = [
    "flash_spectral_clustering",
    "triton_spectral_clustering",
    "cutedsl_spectral_clustering",
    "cutedsl_power_iter_top_k",
    "cutedsl_power_iter_top_k_fused",
    "cutedsl_qmul_eigvecs",
    "cutedsl_qmul_rownorm",
    "cutedsl_row_l2_normalize",
    "cost",
]
