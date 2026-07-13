"""linear_regression dispatcher.

Picks between the Triton (default) and CuteDSL backends.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib.primitives.linear_regression.cutedsl import cutedsl_linear_regression
from flashlib.primitives.linear_regression.triton import triton_linear_regression


def flash_linear_regression(
    X: torch.Tensor,
    y: torch.Tensor,
    n_refine: int = 1,
    *,
    tol: Optional[float] = None,
    backend: Optional[str] = None,
):
    """Linear regression -- exact in input dtype by default.

    ``tol=None`` (default) keeps the dominant ``X.T @ X`` GEMM in input
    dtype (cuBLAS fp32 with iterative refinement); pass ``tol`` to opt
    into a low-precision storage cast.
    """
    if backend == "cutedsl":
        return cutedsl_linear_regression(X, y, n_refine=n_refine, tol=tol or 1e-3)
    return triton_linear_regression(X, y, n_refine=n_refine, tol=tol)
