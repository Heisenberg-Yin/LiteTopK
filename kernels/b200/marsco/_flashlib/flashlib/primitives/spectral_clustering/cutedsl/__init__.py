"""spectral_clustering cutedsl backend (spectral).

Re-exports top-level functions/classes/constants from each component
file. ``@triton.jit`` / ``@cute.jit`` kernels stay private to their
file (call them via the Python wrapper that lives next to them).
"""
from flashlib.primitives.spectral_clustering.cutedsl.spectral import (
    _cache,
    _next_pow2,
    cutedsl_qmul_rownorm,
    cutedsl_qmul_eigvecs,
    cutedsl_row_l2_normalize,
    cutedsl_power_iter_top_k,
    cutedsl_power_iter_top_k_fused,
    cutedsl_spectral_clustering,
)

__all__ = [
    "cutedsl_qmul_rownorm",
    "cutedsl_qmul_eigvecs",
    "cutedsl_row_l2_normalize",
    "cutedsl_power_iter_top_k",
    "cutedsl_power_iter_top_k_fused",
    "cutedsl_spectral_clustering",
]
