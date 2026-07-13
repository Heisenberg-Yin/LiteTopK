"""spectral_clustering triton backend.

Re-exports the public Python wrappers from each component file.
``@triton.jit`` kernels stay private to their file.
"""
from flashlib.primitives.spectral_clustering.triton.spectral import (
    _knn_affinity,
    _knn_normalized_sparse,
    _power_iter_top_k,
    flash_spectral_clustering,
    _kmeans_pp_init_torch,
    _flash_kmeans_with_pp_init,
    triton_spectral_clustering,
    _normalized_similarity_from_knn,
)

__all__ = [
    "flash_spectral_clustering",
    "triton_spectral_clustering",
]
