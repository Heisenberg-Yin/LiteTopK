"""LogisticRegression primitive -- L-BFGS with fused fwd/bwd Triton kernel.

Public API:
    flash_logistic_regression(X, y, *, tol=None, gtol=1e-4, backend=None, ...)
        -> (w, b). Default is **EXACT in input dtype**. Pass ``tol=1e-3``
        to opt into bf16 storage for the dominant GEMVs (routed through
        :func:`flashlib.linalg.gemm.storage_dtype_for`). ``gtol`` is the
        L-BFGS gradient sup-norm convergence threshold (sklearn's
        ``tol`` -- renamed here so the library-wide ``tol`` slot stays
        the precision-tolerance lever).

    triton_logistic_regression          -- explicit Triton entry point
    cutedsl_logistic_regression         -- opt-in CuteDSL fused-fwd path
    flash_cutedsl_logistic_regression   -- alias
    cutedsl_fwd_gemv                    -- standalone CuteDSL fwd GEMV
"""
from flashlib._lazy import lazy_attr
from flashlib.primitives.logistic_regression import cost
from flashlib.primitives.logistic_regression.impl import (
    flash_logistic_regression,
    triton_logistic_regression,
)


cutedsl_logistic_regression = lazy_attr(
    "flashlib.primitives.logistic_regression.cutedsl",
    "cutedsl_logistic_regression",
)
cutedsl_fwd_gemv = lazy_attr(
    "flashlib.primitives.logistic_regression.cutedsl",
    "cutedsl_fwd_gemv",
)
flash_cutedsl_logistic_regression = lazy_attr(
    "flashlib.primitives.logistic_regression.cutedsl",
    "flash_cutedsl_logistic_regression",
)


__all__ = [
    "flash_logistic_regression",
    "triton_logistic_regression",
    "cutedsl_logistic_regression",
    "cutedsl_fwd_gemv",
    "flash_cutedsl_logistic_regression",
    "cost",
]
