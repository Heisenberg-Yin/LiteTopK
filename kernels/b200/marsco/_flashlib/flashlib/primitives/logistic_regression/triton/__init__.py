"""logistic_regression triton backend.

Re-exports the public Python wrappers from each component file.
``@triton.jit`` kernels stay private to their file.
"""
from flashlib.primitives.logistic_regression.triton.logistic_regression import (
    triton_logistic_regression,
    flash_logistic_regression,
    triton_logreg_fwd_bwd,
)

__all__ = [
    "triton_logistic_regression",
    "flash_logistic_regression",
    "triton_logreg_fwd_bwd",
]
