"""Linear algebra primitives — independently callable.

    from flashlib.linalg import cov_gemm, gram_gemm, ab_gemm, eigh, gemm
    from flashlib.linalg.gemm import gemm_fp32, gemm_3xbf16, ...
    from flashlib.linalg.polar import polar, polar_qdwh_hybrid, polar_zolo, ...
    from flashlib.linalg.orthonormalize import cholqr2, split_basis
    from flashlib import cov_gemm, eigh, gemm                # also at top level
"""
from flashlib.linalg.cov_gemm import cov_gemm
from flashlib.linalg.gram_gemm import gram_gemm
from flashlib.linalg.ab_gemm import ab_gemm
# eigh / gemm / polar / orthonormalize are subpackages that expose dispatchers
# plus per-variant primitives. Subpackage import is lazy via __getattr__ to
# avoid eager loading of QDWH (heavy) when only cov_gemm is needed.
from flashlib.linalg import eigh as _eigh_pkg
eigh = _eigh_pkg.eigh

__all__ = ["cov_gemm", "gram_gemm", "ab_gemm", "eigh"]
