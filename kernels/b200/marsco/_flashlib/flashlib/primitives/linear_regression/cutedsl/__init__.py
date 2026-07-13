"""linear_regression cutedsl backend.

Re-exports the public Python wrappers from each component file.
``@cute.jit`` kernels stay private to their file (call them via the
Python wrapper that lives next to them).
"""
from flashlib.primitives.linear_regression.cutedsl.xtx import (
    cutedsl_xtx,
    cutedsl_linear_regression,
    cutedsl_available,
)

__all__ = [
    "cutedsl_xtx",
    "cutedsl_linear_regression",
    "cutedsl_available",
]
