"""linear_regression triton backend.

Re-exports the public Python wrappers from each component file.
``@triton.jit`` kernels stay private to their file.
"""
from flashlib.primitives.linear_regression.triton.linear_regression import (
    triton_linear_regression,
    flash_linear_regression,
)
from flashlib.primitives.linear_regression.triton.fused_kernels import (
    cast_and_xty,
    fused_refine_from_bf16,
)

__all__ = [
    "triton_linear_regression",
    "flash_linear_regression",
    "cast_and_xty",
    "fused_refine_from_bf16",
]
