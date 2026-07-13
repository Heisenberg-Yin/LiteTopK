"""RBF-affinity + row-degree fused kernel (for Spectral Clustering).
"""
import torch
import triton
import triton.language as tl

from flashlib.kernels.distance.triton._common import _round_to_bucket



# =============================================================================
# Kernel 2b: Pairwise L2 with degree accumulation (for Spectral Clustering)
# Fused: compute RBF affinity + accumulate row sums (degree vector)
# =============================================================================

_PAIRWISE_AFFINITY_DEGREE_CONFIGS = [
    triton.Config({"BLOCK_I": 64, "BLOCK_J": 64, "BLOCK_D": 32}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_I": 64, "BLOCK_J": 64, "BLOCK_D": 64}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_I": 128, "BLOCK_J": 64, "BLOCK_D": 32}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_I": 128, "BLOCK_J": 128, "BLOCK_D": 32}, num_stages=1, num_warps=8),
    triton.Config({"BLOCK_I": 64, "BLOCK_J": 128, "BLOCK_D": 64}, num_stages=2, num_warps=8),
]


@triton.autotune(configs=_PAIRWISE_AFFINITY_DEGREE_CONFIGS, key=["N_KEY", "D_KEY"])
@triton.jit
def _pairwise_affinity_degree_kernel(
    X_ptr, X_sq_ptr, W_ptr, DEG_ptr, GAMMA_ptr,
    N, D,
    stride_xn, stride_xd, stride_wn, stride_wm,
    N_KEY: tl.constexpr, D_KEY: tl.constexpr,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute W[i,j] = exp(-gamma * ||xi-xj||^2) and degree[i] = sum_j W[i,j]."""
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
        xi = tl.load(X_ptr + i_offs[:, None] * stride_xn + d_offs[None, :] * stride_xd,
                      mask=i_mask[:, None] & d_mask[None, :], other=0.0)
        xj = tl.load(X_ptr + j_offs[:, None] * stride_xn + d_offs[None, :] * stride_xd,
                      mask=j_mask[:, None] & d_mask[None, :], other=0.0)
        cross += tl.dot(xi, tl.trans(xj))

    dist_sq = xi_sq[:, None] + xj_sq[None, :] - 2.0 * cross
    dist_sq = tl.maximum(dist_sq, 0.0)

    gamma = tl.load(GAMMA_ptr)
    w = tl.exp(-gamma * dist_sq)
    w = tl.where(i_mask[:, None] & j_mask[None, :], w, 0.0)

    # Store affinity
    w_ptrs = W_ptr + i_offs[:, None] * stride_wn + j_offs[None, :] * stride_wm
    tl.store(w_ptrs, w, mask=i_mask[:, None] & j_mask[None, :])

    # Accumulate degree via atomic_add
    row_sum = tl.sum(w, axis=1)
    tl.atomic_add(DEG_ptr + i_offs, row_sum, mask=i_mask)


def triton_affinity_with_degree(X: torch.Tensor, gamma: float, *, tol=None):
    """Compute RBF affinity matrix and degree vector in one pass.

    The fused inner GEMM uses Triton's ``tl.dot`` default precision
    (TF32 for fp32 on Hopper). ``tol`` is accepted for API parity but
    not threaded into the kernel any more -- callers needing strict
    IEEE distances should recompute over the affinity-derived
    candidates with a dedicated kernel.

    Returns:
        W: (N, N) affinity matrix
        degree: (N,) degree vector
    """
    del tol
    assert X.is_cuda and X.ndim == 2
    N, D = X.shape
    X = X.contiguous()

    x_sq = (X * X).sum(dim=1)
    W = torch.empty(N, N, device=X.device, dtype=torch.float32)
    degree = torch.zeros(N, device=X.device, dtype=torch.float32)
    gamma_t = torch.tensor([gamma], device=X.device, dtype=torch.float32)

    D_KEY = _round_to_bucket(D)
    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_I"]),
        triton.cdiv(N, META["BLOCK_J"]),
    )
    _pairwise_affinity_degree_kernel[grid](
        X, x_sq, W, degree, gamma_t,
        N, D,
        X.stride(0), X.stride(1),
        W.stride(0), W.stride(1),
        N_KEY=_round_to_bucket(N), D_KEY=D_KEY,
    )
    return W, degree
