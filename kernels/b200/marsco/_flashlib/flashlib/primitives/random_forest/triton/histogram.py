"""Random Forest histogram split kernel using Triton — one CTA per feature."""

import sys
import os

import torch

from flashlib.kernels.distance.triton._common import _round_to_bucket


def triton_rf_histogram_split(X: torch.Tensor, y: torch.Tensor, n_bins: int = 256):
    """Compute gradient histograms for all features in a single kernel launch.

    Args:
        X: (N, D) float32 raw features
        y: (N,) float32 targets/gradients
        n_bins: number of histogram bins

    Returns:
        hist: (D, n_bins) gradient histogram
        cumsum: (D, n_bins) cumulative sum of histogram
    """
    N, D = X.shape

    # Quantize features into bins
    # Use percentile-based binning
    bin_edges = torch.linspace(0, 1, n_bins + 1, device=X.device)[1:-1]  # n_bins - 1 edges
    # Normalize X to [0, 1] range per feature
    X_min = X.min(dim=0).values
    X_max = X.max(dim=0).values
    X_norm = (X - X_min) / (X_max - X_min + 1e-10)
    X_binned = torch.bucketize(X_norm, bin_edges).int()  # (N, D) values in [0, n_bins-1]

    # Compute histograms using Triton (single launch, one CTA per feature)
    hist = triton_rf_histogram(X_binned, y, D, N_BINS=n_bins)

    # Cumulative sum for split evaluation
    cumsum = torch.cumsum(hist, dim=1)

    return hist, cumsum


# ============================================================================
# RF histogram kernel migrated from kernels/common/triton_kernels.
# ============================================================================

import triton
import triton.language as tl

# =============================================================================
# Kernel 6: Random Forest histogram kernel
# One CTA per feature — streams all N samples, accumulates histogram
# =============================================================================

@triton.jit
def _rf_histogram_kernel(
    X_BIN_ptr,      # (N, D) int32 binned features
    GRAD_ptr,       # (N,) gradients/targets
    HIST_ptr,       # (D, N_BINS) output histogram
    N,              # regular param
    D: tl.constexpr,
    N_BINS: tl.constexpr,
    stride_xn, stride_xd,
    stride_hd, stride_hb,
    N_KEY: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One CTA per feature: vectorized atomic scatter to histogram."""
    pid_d = tl.program_id(0)  # feature index
    if pid_d >= D:
        return

    hist_base = HIST_ptr + pid_d * stride_hd

    for n_start in tl.range(0, N_KEY, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # Vectorized loads — all threads load in parallel
        bins = tl.load(X_BIN_ptr + n_offs * stride_xn + pid_d * stride_xd,
                       mask=n_mask, other=0)
        grads = tl.load(GRAD_ptr + n_offs, mask=n_mask, other=0.0)

        # Vectorized atomic scatter — each thread atomically adds to its bin
        tl.atomic_add(hist_base + bins * stride_hb, grads, mask=n_mask)


def triton_rf_histogram(X_binned, gradients, D, N_BINS=256, BLOCK_N=512):
    """Compute gradient histograms for all features in one launch.

    Args:
        X_binned: (N, D) int32 binned features
        gradients: (N,) float32 gradient values
        D: number of features

    Returns:
        hist: (D, N_BINS) histogram
    """
    N = X_binned.shape[0]
    hist = torch.zeros(D, N_BINS, device=X_binned.device, dtype=torch.float32)

    grid = (D,)
    _rf_histogram_kernel[grid](
        X_binned, gradients, hist,
        N, D, N_BINS,
        X_binned.stride(0), X_binned.stride(1),
        hist.stride(0), hist.stride(1),
        N_KEY=_round_to_bucket(N),
        BLOCK_N=BLOCK_N,
    )
    return hist

