"""distance triton backend (affinity, mrd, pairwise).

Re-exports top-level functions/classes/constants from each component
file. ``@triton.jit`` / ``@cute.jit`` kernels stay private to their
file (call them via the Python wrapper that lives next to them).
"""
from flashlib.kernels.distance.triton.affinity import (
    triton_affinity_with_degree,
)
from flashlib.kernels.distance.triton.mrd import (
    triton_pairwise_mrd,
    triton_fused_mrd_edges,
)
from flashlib.kernels.distance.triton.pairwise import (
    _round_to_bucket,
    _CONFIGS,
    pairwise_l2sq,
    pairwise_l2,
    triton_rbf_kernel,
)
from flashlib.kernels.distance.triton.knn_gather_l2sq import triton_knn_gather_sqdist

__all__ = [
    "triton_affinity_with_degree",
    "triton_pairwise_mrd",
    "triton_fused_mrd_edges",
    "pairwise_l2sq",
    "pairwise_l2",
    "triton_rbf_kernel",
    "triton_knn_gather_sqdist",
]
