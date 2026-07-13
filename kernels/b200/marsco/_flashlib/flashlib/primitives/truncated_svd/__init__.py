"""TruncatedSVD primitive -- exact in input dtype by default.

Public API:
    flash_truncated_svd(X, K, *, tol=None, backend=None) -> (S, Vh)

``tol=None`` (default) routes to the Triton path and runs the exact
eigh on the cov / Gram matrix (cuSOLVER / MKL). Pass ``tol >= 1e-3`` on
wide / square shapes to opt into the CuteDSL bf16-fused path; pass
``tol >= 1e-4`` on tall shapes to opt into Halko subspace iteration via
``flashlib.linalg.eigh``.
"""
from flashlib._lazy import lazy_attr
from flashlib.primitives.truncated_svd import cost
from flashlib.primitives.truncated_svd.impl import (
    flash_truncated_svd,
    triton_truncated_svd,
)


cutedsl_truncated_svd = lazy_attr(
    "flashlib.primitives.truncated_svd.cutedsl",
    "cutedsl_truncated_svd",
)
cutedsl_cov_gemm = lazy_attr(
    "flashlib.primitives.truncated_svd.cutedsl",
    "cutedsl_cov_gemm",
)
cutedsl_gram_gemm = lazy_attr(
    "flashlib.primitives.truncated_svd.cutedsl",
    "cutedsl_gram_gemm",
)
cutedsl_cov_gemm_simt = lazy_attr(
    "flashlib.primitives.truncated_svd.cutedsl",
    "cutedsl_cov_gemm_simt",
)
cutedsl_gram_gemm_simt = lazy_attr(
    "flashlib.primitives.truncated_svd.cutedsl",
    "cutedsl_gram_gemm_simt",
)
flash_truncated_svd_cutedsl = lazy_attr(
    "flashlib.primitives.truncated_svd.cutedsl",
    "flash_truncated_svd_cutedsl",
)


__all__ = [
    "flash_truncated_svd",
    "triton_truncated_svd",
    "cutedsl_truncated_svd",
    "cutedsl_cov_gemm",
    "cutedsl_gram_gemm",
    "cutedsl_cov_gemm_simt",
    "cutedsl_gram_gemm_simt",
    "flash_truncated_svd_cutedsl",
    "cost",
]
