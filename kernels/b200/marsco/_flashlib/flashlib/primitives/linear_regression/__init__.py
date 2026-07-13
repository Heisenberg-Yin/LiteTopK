"""LinearRegression primitive -- normal equations via cov_gemm + Cholesky.

The Triton entry point uses :func:`flashlib.linalg.gemm.gemm` with
``tol=1e-3`` (cuBLAS bf16 tensor cores + fp32 accumulator) plus mixed
precision iterative refinement. The CuteDSL entry point shares the
fused cast/xty/refine kernels and routes the dominant ``X.T @ X`` GEMM
through :mod:`flashlib.linalg.gemm` as well.
"""
from flashlib._lazy import lazy_attr
from flashlib.primitives.linear_regression import cost
from flashlib.primitives.linear_regression.impl import flash_linear_regression
from flashlib.primitives.linear_regression.triton import (
    triton_linear_regression,
)

cutedsl_linear_regression = lazy_attr(
    "flashlib.primitives.linear_regression.cutedsl",
    "cutedsl_linear_regression",
)
cutedsl_xtx = lazy_attr(
    "flashlib.primitives.linear_regression.cutedsl",
    "cutedsl_xtx",
)


__all__ = [
    "flash_linear_regression",
    "triton_linear_regression",
    "cutedsl_linear_regression",
    "cutedsl_xtx",
    "cost",
]
