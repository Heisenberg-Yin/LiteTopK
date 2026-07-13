"""linalg.cholesky — Cholesky factorization (A = L Lᵀ for SPD A).

Variants:
    potrf_3xbf16(A)   — Block-recursive POTRF with 3xBF16 trailing SYRK
                        (CuTeDSL backend). Trailing SYRK dominates FLOPs
                        and is routed through 3xBF16; leaf POTRF stays
                        fp32 via cuSOLVER.
"""
from flashlib.linalg.cholesky.cutedsl import potrf_3xbf16

__all__ = ["potrf_3xbf16"]
