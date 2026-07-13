"""cov_gemm triton backend (cov_gemm).

Re-exports top-level functions/classes/constants from each component
file. ``@triton.jit`` / ``@cute.jit`` kernels stay private to their
file (call them via the Python wrapper that lives next to them).
"""
from flashlib.linalg.cov_gemm.triton.cov_gemm import (
    _round_to_bucket,
    _CONFIGS,
    cov_gemm,
    full_gemm,
)

__all__ = [
    "cov_gemm",
    "full_gemm",
]
