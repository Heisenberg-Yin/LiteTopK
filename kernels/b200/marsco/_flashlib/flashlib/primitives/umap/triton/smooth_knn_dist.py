"""UMAP fuzzy simplicial set Stage 2: smooth k-NN distance.

One Triton program per row, 64 bisection iterations to find sigma
s.t. sum_j exp(-(d_ij - rho_i)/sigma_i) ~= log2(k).
"""
import math
import numpy as np
import torch
import triton
import triton.language as tl



# =============================================================================
# Kernel: Smooth-knn distance (UMAP fuzzy simplicial set Stage 2)
# One program per row, 64 bisection iterations to find sigma s.t.
#   sum_j exp(-(d_ij - rho_i)/sigma_i) ~= log2(k)
# =============================================================================

@triton.jit
def _smooth_knn_dist_kernel(
    DIST_ptr,      # (N, K) sorted neighbor distances (float32)
    SIGMA_ptr,     # (N,) output
    RHO_ptr,       # (N,) output
    N: tl.constexpr,
    K: tl.constexpr,
    TARGET,        # log2(K) * bandwidth (float32 runtime)
    N_ITER: tl.constexpr,
    TOL: tl.constexpr,
    BLOCK_K: tl.constexpr,  # next_pow2(K)
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K

    # Load row of distances
    d_row = tl.load(DIST_ptr + pid * K + k_offs, mask=k_mask, other=0.0)

    # rho = first non-zero distance (UMAP local_connectivity=1.0 case)
    nonzero = (d_row > 0.0) & k_mask
    has_nonzero = tl.sum(nonzero.to(tl.int32)) > 0
    very_large = 1e30
    masked = tl.where(nonzero, d_row, very_large)
    rho = tl.min(masked)
    rho = tl.where(has_nonzero, rho, 0.0)

    # Bisection on sigma (mid)
    lo = 0.0
    hi = 1e30
    mid = 1.0
    for _ in tl.static_range(N_ITER):
        # psum = sum over j>=1 of exp(-(d_ij - rho)/mid) (or 1 if d <= rho)
        # j=0 (self) excluded since d_row[0]=0 typically => d-rho<=0 => contributes 1
        # Upstream UMAP starts j from 1, but contribution is the same in either case
        diff = d_row - rho
        contrib = tl.where(diff > 0, tl.exp(-diff / mid), 1.0)
        # exclude j=0 (the self distance)
        contrib = tl.where(k_offs == 0, 0.0, contrib)
        contrib = tl.where(k_mask, contrib, 0.0)
        psum = tl.sum(contrib)

        # Update bounds
        too_high = psum > TARGET
        # if too_high: hi = mid; mid = (lo + hi)/2
        # else:        lo = mid; mid = mid*2 if hi == inf else (lo+hi)/2
        new_hi_too_high = mid
        new_mid_too_high = (lo + new_hi_too_high) * 0.5

        new_lo_too_low = mid
        # if hi >= 1e29: mid = mid * 2 else mid = (lo+hi)/2
        hi_is_inf = hi >= 1e29
        new_mid_too_low = tl.where(hi_is_inf, mid * 2.0, (new_lo_too_low + hi) * 0.5)

        # Apply (gate by tolerance)
        within = tl.abs(psum - TARGET) < TOL
        # When converged we keep current mid; emulate "break" by no-op
        new_lo = tl.where(within, lo, tl.where(too_high, lo, new_lo_too_low))
        new_hi = tl.where(within, hi, tl.where(too_high, new_hi_too_high, hi))
        new_mid = tl.where(within, mid, tl.where(too_high, new_mid_too_high, new_mid_too_low))
        lo = new_lo
        hi = new_hi
        mid = new_mid

    tl.store(SIGMA_ptr + pid, mid)
    tl.store(RHO_ptr + pid, rho)


def triton_smooth_knn_dist(distances: torch.Tensor, n_iter: int = 64,
                           bandwidth: float = 1.0, tol: float = 1e-5):
    """Compute (sigma, rho) per row for UMAP fuzzy simplicial set.

    Args:
        distances: (N, K) float32 tensor of sorted distances to k nearest neighbors.
                   distances[:, 0] should be 0 (self).
        n_iter: number of bisection iterations.
        bandwidth: UMAP bandwidth parameter.
        tol: convergence tolerance.

    Returns:
        sigma: (N,) float32 — bandwidth parameter per point
        rho: (N,) float32 — distance to nearest non-self neighbor per point
    """
    N, K = distances.shape
    assert distances.dtype == torch.float32 and distances.is_cuda
    sigma = torch.empty(N, device=distances.device, dtype=torch.float32)
    rho = torch.empty(N, device=distances.device, dtype=torch.float32)
    target = float(np.log2(K) * bandwidth)
    # next power-of-two >= K, minimum 16
    BLOCK_K = max(16, 1 << (K - 1).bit_length())
    grid = (N,)
    _smooth_knn_dist_kernel[grid](
        distances, sigma, rho,
        N=N, K=K, TARGET=target,
        N_ITER=n_iter, TOL=tol, BLOCK_K=BLOCK_K,
        num_warps=1,
    )
    return sigma, rho
