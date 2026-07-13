"""Ridge regression primitive — bf16 normal equations + alpha-aware refinement.

Public API:
    flash_ridge(X, y, alpha=1.0, *, tol=1e-3, n_refine=1, backend=None)
    flash_ridge_regression                — alias, same signature
    cutedsl_ridge_regression              — CuteDSL bf16 symmetric XtX path
                                            (lazy; opt-in via backend="cutedsl")
"""
from flashlib._lazy import lazy_attr
from flashlib.primitives.ridge import cost
from flashlib.primitives.ridge.impl import (
    flash_ridge,
    flash_ridge_regression,
)
from flashlib.primitives.ridge.triton import triton_ridge_regression


cutedsl_ridge_regression = lazy_attr(
    "flashlib.primitives.ridge.cutedsl", "cutedsl_ridge_regression",
)
cutedsl_available = lazy_attr(
    "flashlib.primitives.ridge.cutedsl", "cutedsl_available",
)


__all__ = [
    "flash_ridge",
    "flash_ridge_regression",
    "triton_ridge_regression",
    "cutedsl_ridge_regression",
    "cutedsl_available",
    "cost",
]
