"""Legacy ridge regression — triton_cov_gemm + LU solve (kept for parity).

This is the pre-R3 path: TF32 normal-equation GEMM via the in-house
``triton_cov_gemm`` (symmetric, ~2x fewer FLOPs vs cuBLAS generic) plus
a tiny cuSOLVER LU solve. Superseded by :func:`triton_ridge_regression`
in [ridge.py](ridge.py) which switches the dominant GEMM to bf16 + adds
alpha-aware iterative refinement (~7x faster, 2 OOM tighter weights).

Kept under ``triton/`` because it is still a Triton-backed implementation
(uses ``triton_cov_gemm``) — only the dispatch entry point moved.
"""
import torch

from flashlib.linalg.cov_gemm import cov_gemm as triton_cov_gemm


def triton_ridge_regression_legacy(X: torch.Tensor, y: torch.Tensor,
                                    alpha: float = 1.0):
    """Legacy ridge: TF32 cov_gemm + LU solve.

    Args:
        X: (N, D) float32
        y: (N,) float32
        alpha: regularisation strength

    Returns:
        w: (D,) weight vector
    """
    XtX = triton_cov_gemm(X)
    Xty = X.T @ y
    XtX.diagonal().add_(alpha)
    return torch.linalg.solve(XtX, Xty)
