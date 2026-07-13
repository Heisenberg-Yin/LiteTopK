"""gram_gemm triton backend (gram_gemm).

Re-exports top-level functions/classes/constants from each component
file. ``@triton.jit`` / ``@cute.jit`` kernels stay private to their
file (call them via the Python wrapper that lives next to them).
"""
from flashlib.linalg.gram_gemm.triton.gram_gemm import (
    _round_to_bucket,
    _CONFIGS,
    gram_gemm,
)

__all__ = [
    "gram_gemm",
]
