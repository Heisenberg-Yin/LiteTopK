"""Pairwise L2 distance kernels with fused output transforms.

Uses ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x.y (pre-compute norms, tl.dot for cross).
FUSE_OP constexpr selects: 0 = squared, 1 = RBF exp(-gamma*d^2), 2 = sqrt.

Serves: DBSCAN, HDBSCAN, KNN-naive baseline, SVC, SpectralClust, KDE.
"""
import math
import torch
import triton
import triton.language as tl

from flashlib.kernels.distance.triton._common import _round_to_bucket



_CONFIGS = [
    triton.Config({"BLOCK_I": 64, "BLOCK_J": 64, "BLOCK_D": 32}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_I": 64, "BLOCK_J": 64, "BLOCK_D": 64}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_I": 128, "BLOCK_J": 64, "BLOCK_D": 32}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_I": 128, "BLOCK_J": 128, "BLOCK_D": 32}, num_stages=1, num_warps=8),
    triton.Config({"BLOCK_I": 64, "BLOCK_J": 128, "BLOCK_D": 64}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_I": 32, "BLOCK_J": 32, "BLOCK_D": 64}, num_stages=2, num_warps=4),
]


@triton.autotune(configs=_CONFIGS, key=["N_KEY", "D_KEY"])
@triton.jit
def _pairwise_l2_kernel(
    X_ptr, X_sq_ptr, OUT_ptr, GAMMA_ptr,
    N, D,
    stride_xn, stride_xd,
    stride_on, stride_om,
    N_KEY: tl.constexpr,
    D_KEY: tl.constexpr,
    FUSE_OP: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    i_offs = (pid_i * BLOCK_I + tl.arange(0, BLOCK_I)).to(tl.int64)
    j_offs = (pid_j * BLOCK_J + tl.arange(0, BLOCK_J)).to(tl.int64)
    i_mask = i_offs < N
    j_mask = j_offs < N

    xi_sq = tl.load(X_sq_ptr + i_offs, mask=i_mask, other=0.0)
    xj_sq = tl.load(X_sq_ptr + j_offs, mask=j_mask, other=0.0)

    cross = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
    for d_start in tl.range(0, D_KEY, BLOCK_D, num_stages=2):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        xi_ptrs = X_ptr + i_offs[:, None] * stride_xn + d_offs[None, :] * stride_xd
        xi = tl.load(xi_ptrs, mask=i_mask[:, None] & d_mask[None, :], other=0.0)

        xj_ptrs = X_ptr + j_offs[:, None] * stride_xn + d_offs[None, :] * stride_xd
        xj = tl.load(xj_ptrs, mask=j_mask[:, None] & d_mask[None, :], other=0.0)

        cross += tl.dot(xi, tl.trans(xj))

    dist_sq = xi_sq[:, None] + xj_sq[None, :] - 2.0 * cross
    dist_sq = tl.maximum(dist_sq, 0.0)

    if FUSE_OP == 0:
        result = dist_sq
    elif FUSE_OP == 1:
        gamma = tl.load(GAMMA_ptr)
        result = tl.exp(-gamma * dist_sq)
    elif FUSE_OP == 2:
        result = tl.sqrt(dist_sq)
    else:
        result = dist_sq

    out_ptrs = OUT_ptr + i_offs[:, None] * stride_on + j_offs[None, :] * stride_om
    tl.store(out_ptrs, result, mask=i_mask[:, None] & j_mask[None, :])


def pairwise_l2sq(X: torch.Tensor, *, tol=None) -> torch.Tensor:
    """Pairwise squared Euclidean distance: out[i, j] = ||X[i] - X[j]||^2.

    Args:
        X: (N, D) CUDA tensor (any float dtype).
        tol: kept for API compatibility; the inner GEMM uses the
            triton ``tl.dot`` default precision (TF32 on Hopper for
            fp32 inputs). Distances are accumulated in fp32. Callers
            that need strict-IEEE distances should recompute over
            the returned candidate indices with a dedicated kernel.

    Returns:
        (N, N) float32 squared distance matrix.
    """
    del tol
    assert X.is_cuda and X.ndim == 2
    N, D = X.shape
    X = X.contiguous()

    x_sq = (X * X).sum(dim=1)
    out = torch.empty(N, N, device=X.device, dtype=torch.float32)
    gamma_dummy = torch.zeros(1, device=X.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(N, META["BLOCK_I"]), triton.cdiv(N, META["BLOCK_J"]))
    _pairwise_l2_kernel[grid](
        X, x_sq, out, gamma_dummy,
        N, D,
        X.stride(0), X.stride(1),
        out.stride(0), out.stride(1),
        N_KEY=_round_to_bucket(N), D_KEY=_round_to_bucket(D),
        FUSE_OP=0,
    )
    return out


def pairwise_l2(X: torch.Tensor, *, tol=None) -> torch.Tensor:
    """Pairwise Euclidean distance: out[i, j] = ||X[i] - X[j]|| (with sqrt)."""
    del tol
    assert X.is_cuda and X.ndim == 2
    N, D = X.shape
    X = X.contiguous()

    x_sq = (X * X).sum(dim=1)
    out = torch.empty(N, N, device=X.device, dtype=torch.float32)
    gamma_dummy = torch.zeros(1, device=X.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(N, META["BLOCK_I"]), triton.cdiv(N, META["BLOCK_J"]))
    _pairwise_l2_kernel[grid](
        X, x_sq, out, gamma_dummy,
        N, D,
        X.stride(0), X.stride(1),
        out.stride(0), out.stride(1),
        N_KEY=_round_to_bucket(N), D_KEY=_round_to_bucket(D),
        FUSE_OP=2,
    )
    return out


# ============================================================================
# Additional pairwise-distance kernels migrated from kernels/common/.
# RBF affinity, degree-fused affinity, mutual-reachability distance (HDBSCAN),
# and fused KNN-edge MRD reduction.
# ============================================================================

import numpy as np

def triton_rbf_kernel(X: torch.Tensor, gamma: float, *, tol=None) -> torch.Tensor:
    """Compute RBF kernel matrix: K[i,j] = exp(-gamma * ||x_i - x_j||^2).

    Args:
        X: (N, D) CUDA tensor (any float dtype).
        gamma: RBF kernel parameter.
        tol: kept for API compatibility (see :func:`pairwise_l2sq`).

    Returns:
        (N, N) float32 kernel matrix
    """
    del tol
    assert X.is_cuda and X.ndim == 2
    N, D = X.shape
    X = X.contiguous()

    x_sq = (X * X).sum(dim=1)
    out = torch.empty(N, N, device=X.device, dtype=torch.float32)
    gamma_t = torch.tensor([gamma], device=X.device, dtype=torch.float32)

    D_KEY = _round_to_bucket(D)
    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_I"]),
        triton.cdiv(N, META["BLOCK_J"]),
    )
    _pairwise_l2_kernel[grid](
        X, x_sq, out, gamma_t,
        N, D,
        X.stride(0), X.stride(1),
        out.stride(0), out.stride(1),
        N_KEY=_round_to_bucket(N), D_KEY=D_KEY,
        FUSE_OP=1,
    )
    return out
