"""linalg.qr — QR factorization (Q, R).

Variants:
    geqrf_3xbf16(A)   — CholeskyQR2 with 3xBF16 GEMM/TRSM (CuTeDSL backend).
                        For κ(A) ≲ u⁻¹ᐟ² ≈ 2·10³ this matches Householder QR
                        to the precision floor; a third pass is auto-added if
                        orthogonality residual overruns the budget.
"""
from flashlib.linalg.qr.cutedsl import geqrf_3xbf16

__all__ = ["geqrf_3xbf16"]
