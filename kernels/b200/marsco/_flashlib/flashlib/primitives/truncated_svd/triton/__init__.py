"""truncated_svd triton backend.

Re-exports the public Python wrappers from each component file.
``@triton.jit`` kernels stay private to their file (call them via the
Python wrapper that lives next to them).
"""
from flashlib.primitives.truncated_svd.triton.svd import (
    _triton_svd_cov,
    _triton_svd_dual,
    triton_truncated_svd,
    flash_truncated_svd,
)
from flashlib.primitives.truncated_svd.triton.fused_kernels import (
    cublas_bf16_cov_gemm,
    cublas_bf16_gram_gemm,
    subspace_iteration_eigh,
    fused_vproj_norm_to_vh,
)

__all__ = [
    "triton_truncated_svd",
    "flash_truncated_svd",
    "cublas_bf16_cov_gemm",
    "cublas_bf16_gram_gemm",
    "subspace_iteration_eigh",
    "fused_vproj_norm_to_vh",
]
