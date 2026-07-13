"""pca triton backend.

Re-exports the public Python wrappers from each component file.
``@triton.jit`` kernels stay private to their file (call them via the
Python wrapper that lives next to them).
"""
from flashlib.primitives.pca.triton.pca import (
    _triton_pca_cov,
    _triton_pca_dual,
    triton_pca,
    flash_pca,
)
from flashlib.primitives.pca.triton.fused_kernels import (
    triton_cov_gemm_fused,
    triton_gram_gemm_fused,
    triton_eigh_upper,
)

__all__ = [
    "triton_pca",
    "flash_pca",
    "triton_cov_gemm_fused",
    "triton_gram_gemm_fused",
    "triton_eigh_upper",
]
