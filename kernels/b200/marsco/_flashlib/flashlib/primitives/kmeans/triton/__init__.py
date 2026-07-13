"""kmeans triton backend.

Re-exports the public Python wrappers from each component file.
``@triton.jit`` kernels stay private to their file.
"""
from flashlib.primitives.kmeans.triton.assign import (
    _ceil_div,
    euclid_assign_triton,
    cosine_assign_triton,
)
from flashlib.primitives.kmeans.triton.kmeans import (
    COMPILE_FLAG,
    batch_kmeans_Euclid,
    batch_kmeans_Cosine,
    batch_kmeans_Dot,
)
from flashlib.primitives.kmeans.triton.update import (
    triton_centroid_update_cosine,
    torch_loop_centroid_update_cosine,
    triton_centroid_update_euclid,
    triton_centroid_update_sorted_cosine,
    triton_centroid_update_sorted_euclid,
    triton_centroid_finalize,
    triton_lloyd_centroid_step_euclid,
    main,
)

__all__ = [
    "euclid_assign_triton",
    "cosine_assign_triton",
    "COMPILE_FLAG",
    "batch_kmeans_Euclid",
    "batch_kmeans_Cosine",
    "batch_kmeans_Dot",
    "triton_centroid_update_cosine",
    "torch_loop_centroid_update_cosine",
    "triton_centroid_update_euclid",
    "triton_centroid_update_sorted_cosine",
    "triton_centroid_update_sorted_euclid",
    "triton_centroid_finalize",
    "triton_lloyd_centroid_step_euclid",
    "main",
]
