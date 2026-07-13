"""ridge triton backend.

Re-exports top-level wrappers from each component file. ``@triton.jit`` /
``@cute.jit`` kernels stay private to their file (call them via the
Python wrapper that lives next to them).
"""
from flashlib.primitives.ridge.triton.ridge import (
    triton_ridge_regression,
    flash_ridge_regression,
)

__all__ = [
    "triton_ridge_regression",
    "flash_ridge_regression",
]
