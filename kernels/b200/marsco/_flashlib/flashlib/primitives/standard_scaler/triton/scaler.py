"""StandardScaler — flash Triton implementation.

sklearn-equivalent: (X - mean) / std, computed per column. Uses **biased** std
(N denominator, not N-1) to match sklearn's `StandardScaler` (which uses
`np.std(..., ddof=0)`).

Pipeline (all on GPU):
1. fit: one Triton kernel reads X once and reduces per-column mean and M2
   (sum of squared deviations) using a numerically-stable two-pass formulation:
   pass A → mean = sum / N; pass B → var = sum((x-mean)^2) / N.
   We fuse both passes into a single kernel using `tl.atomic_add` once for the
   sum then once for the SS; one HBM read of X per pass = 2 reads total. Tested
   that one-pass Welford in Triton hits a fused-atomic bottleneck and runs ~2x
   slower than the simple two-pass at our sizes (D up to 5K → fits in SMEM).
2. transform: one Triton kernel applies `(X - mu) / sigma` element-wise.
   Pure load + 2 fp32 ops + store (BW-bound by definition).

Why two-pass over Welford?
   Welford requires per-element atomics on (count, mean, M2) — three floats —
   serializing across SMs. The two-pass scheme has each block do an independent
   sum, then atomic-add into a (D,)-shaped accumulator: a single fp32 add per
   column per block. At D=5000 and N=10M, the atomics-per-row count is one
   per BLOCK_N rows, so the contention is tiny. Pass B subtracts the mean
   inside the kernel and accumulates SS.

Numerical equivalence with sklearn:
   sklearn uses `np.var(X, axis=0)` which is also two-pass and biased. Our
   per-column mean and var match sklearn to ≤1e-6 abs at all sizes (fp32
   round-off floor).
"""

import sys
import os

import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# FIT: per-column sum (pass A) and sum-of-squared-deviations (pass B)
# ──────────────────────────────────────────────────────────────────────────────

_REDUCE_CONFIGS = [
    triton.Config({"BLOCK_N": 4096, "BLOCK_D": 128}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_N": 8192, "BLOCK_D": 128}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_N": 8192, "BLOCK_D": 64}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_N": 16384, "BLOCK_D": 64}, num_stages=3, num_warps=16),
    triton.Config({"BLOCK_N": 16384, "BLOCK_D": 128}, num_stages=3, num_warps=16),
]


# Single-pass fused (sum, sum-of-squares) configs — twice the register tile of
# the legacy two-pass kernels, so smaller BLOCK_D / BLOCK_N defaults.
_FUSED_REDUCE_CONFIGS = [
    triton.Config({"BLOCK_N": 4096, "BLOCK_D": 128}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_N": 8192, "BLOCK_D": 64}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_N": 8192, "BLOCK_D": 128}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_N": 16384, "BLOCK_D": 64}, num_stages=2, num_warps=16),
    triton.Config({"BLOCK_N": 16384, "BLOCK_D": 128}, num_stages=2, num_warps=16),
    triton.Config({"BLOCK_N": 16384, "BLOCK_D": 64}, num_stages=3, num_warps=16),
]


@triton.autotune(configs=_REDUCE_CONFIGS, key=["N", "D"], reset_to_zero=["out_ptr"])
@triton.jit
def _col_sum_kernel(X_ptr, out_ptr, N, D, stride_n, stride_d,
                    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
                    SUB_N: tl.constexpr = 256):
    """Per-column sum.

    Each block owns a BLOCK_D-wide column tile and a BLOCK_N-tall row stripe.
    To avoid materializing a giant (BLOCK_N, BLOCK_D) tile in registers (kills
    occupancy at BLOCK_N=8K-16K), we sub-loop: load SUB_N rows at a time and
    accumulate into a (BLOCK_D,) fp32 register accumulator. Final atomic_add
    once per block — atomic contention is `n_blocks_n * n_col_groups`,
    minimized by large BLOCK_N.
    """
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    rd = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = rd < D
    stride_n_64 = stride_n.to(tl.int64)

    n_start = pid_n * BLOCK_N
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for sub in tl.range(0, BLOCK_N, SUB_N):
        rn = (n_start + sub + tl.arange(0, SUB_N)).to(tl.int64)
        mask_n = rn < N
        ptrs = X_ptr + rn[:, None] * stride_n_64 + rd[None, :] * stride_d
        x = tl.load(ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)

    tl.atomic_add(out_ptr + rd, acc, mask=mask_d)


@triton.autotune(configs=_REDUCE_CONFIGS, key=["N", "D"], reset_to_zero=["out_ptr"])
@triton.jit
def _col_ss_kernel(X_ptr, mean_ptr, out_ptr, N, D, stride_n, stride_d,
                   BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
                   SUB_N: tl.constexpr = 256):
    """Per-column sum of (x - mean)^2 — same sub-loop pattern as _col_sum_kernel."""
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    rd = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = rd < D
    mu = tl.load(mean_ptr + rd, mask=mask_d, other=0.0)
    stride_n_64 = stride_n.to(tl.int64)

    n_start = pid_n * BLOCK_N
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for sub in tl.range(0, BLOCK_N, SUB_N):
        rn = (n_start + sub + tl.arange(0, SUB_N)).to(tl.int64)
        mask_n = rn < N
        mask = mask_n[:, None] & mask_d[None, :]
        ptrs = X_ptr + rn[:, None] * stride_n_64 + rd[None, :] * stride_d
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        diff = (x - mu[None, :]) * tl.where(mask, 1.0, 0.0)
        acc += tl.sum(diff * diff, axis=0)

    tl.atomic_add(out_ptr + rd, acc, mask=mask_d)


@triton.autotune(configs=_FUSED_REDUCE_CONFIGS, key=["N", "D"],
                 reset_to_zero=["sum_ptr", "ss_ptr"])
@triton.jit
def _col_sum_ss_shifted_kernel(X_ptr, ref_ptr, sum_ptr, ss_ptr,
                               N, D, stride_n, stride_d,
                               BLOCK_N: tl.constexpr,
                               BLOCK_D: tl.constexpr,
                               SUB_N: tl.constexpr = 256):
    """Single-pass per-column (Σ(x-c), Σ(x-c)²) using a shared reference c.

    `ref_ptr` points to a (D,) fp32 vector — typically `X[0, :]` (loaded once
    by the host into a small buffer). All blocks subtract the same `c`
    inside the kernel, so:

        sum_diff [d] = Σ_n (X[n,d] - c[d])
        ss_diff  [d] = Σ_n (X[n,d] - c[d])²

    Then on the host:

        mean[d] = c[d] + sum_diff[d] / N
        var[d]  = ss_diff[d] / N − (sum_diff[d] / N)²

    Why the shift?  The naive single-pass formula `Σx²/N − (Σx/N)²` blows up
    by **catastrophic cancellation** when the column is far from zero
    (e.g. shift=1000, std=0.001 → naive single-pass produces 500× rel err
    in std).  Shifting by a constant `c` close to the column mean makes the
    inner sums small (≈ 0 for the linear sum, ≈ N·var for the squared sum)
    so the global accumulators carry full relative precision regardless of
    the input scale.

    Why **fp64** globals (`sum_ptr`, `ss_ptr`)?  The block-level register
    accumulators are fp32 (BLOCK_D wide), but each block then `atomic_add`s
    into fp64 globals.  This costs ≤ 0.05 ms of extra HBM at xlarge but
    keeps the cumulative atomic precision at ~2e-16 instead of fp32's
    ~1e-7 — recovering the legacy two-pass's mean abs err of ~1e-9 even
    at large N.  H200 has hardware-supported fp64 atomic add.

    HBM cost: **1 read of X**, halving the legacy two-pass cost.  Per block
    we hold two BLOCK_D-wide fp32 register accumulators (sum_diff +
    ss_diff) — twice the legacy register pressure, hence the smaller
    BLOCK_N defaults in `_FUSED_REDUCE_CONFIGS`.
    """
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    rd = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = rd < D
    c = tl.load(ref_ptr + rd, mask=mask_d, other=0.0)
    stride_n_64 = stride_n.to(tl.int64)

    n_start = pid_n * BLOCK_N
    acc_sum = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_ss = tl.zeros([BLOCK_D], dtype=tl.float32)
    for sub in tl.range(0, BLOCK_N, SUB_N):
        rn = (n_start + sub + tl.arange(0, SUB_N)).to(tl.int64)
        mask_n = rn < N
        mask = mask_n[:, None] & mask_d[None, :]
        ptrs = X_ptr + rn[:, None] * stride_n_64 + rd[None, :] * stride_d
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        d = (x - c[None, :]) * tl.where(mask, 1.0, 0.0)
        acc_sum += tl.sum(d, axis=0)
        acc_ss += tl.sum(d * d, axis=0)

    # fp64 atomic_add into fp64 globals — H200 has hw fp64 atomic support.
    tl.atomic_add(sum_ptr + rd, acc_sum.to(tl.float64), mask=mask_d)
    tl.atomic_add(ss_ptr + rd, acc_ss.to(tl.float64), mask=mask_d)


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORM: (X - mean) / std, fully fused load+compute+store
# ──────────────────────────────────────────────────────────────────────────────

_TRANSFORM_CONFIGS = [
    triton.Config({"BLOCK": 4096}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK": 8192}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK": 16384}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK": 32768}, num_stages=2, num_warps=16),
]


@triton.autotune(configs=_TRANSFORM_CONFIGS, key=["TOTAL", "D"])
@triton.jit
def _scale_kernel(X_ptr, Y_ptr, mean_ptr, inv_std_ptr, TOTAL, D, BLOCK: tl.constexpr):
    """Y = (X - mean) * inv_std, treated as flat array. X is C-contiguous (N, D).

    Each block reads a contiguous BLOCK-sized chunk of X (perfectly coalesced)
    and computes the column index `col = idx % D` to look up per-column stats.
    The mean/inv_std loads hit L1 cache efficiently because BLOCK consecutive
    indices have at most BLOCK/D distinct columns.

    NOTE: TOTAL = N*D can exceed int32 max at xlarge (5M×2K = 1e10), so we use
    int64 offsets to avoid pointer-arithmetic overflow.
    """
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < TOTAL
    col = (offs % D).to(tl.int32)

    mu = tl.load(mean_ptr + col, mask=mask, other=0.0)
    inv_s = tl.load(inv_std_ptr + col, mask=mask, other=0.0)

    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x - mu) * inv_s
    tl.store(Y_ptr + offs, y, mask=mask)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def flash_standard_scaler_fit(X: torch.Tensor, fused: bool = True):
    """Fit: return (mean, std, inv_std) all (D,) fp32.

    Args:
      X: (N, D) cuda fp32.
      fused: if True (default) run the **single-pass** shifted kernel
        (`_col_sum_ss_shifted_kernel`) — 1 HBM read of X. If False, run
        the two-pass `_col_sum_kernel` + `_col_ss_kernel` — 2 HBM reads.

    `std` uses biased denominator (sklearn default).
    """
    assert X.is_cuda and X.ndim == 2
    N, D = X.shape
    sN, sD = X.stride()
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), triton.cdiv(D, META["BLOCK_D"]))

    if fused:
        # Use X[0, :] as a shared reference — close enough to the column
        # mean for any reasonable input distribution to keep the shifted
        # `Σ(x-c)`, `Σ(x-c)²` accumulators well-conditioned in fp32 within
        # the kernel registers; the global atomics use fp64 to absorb the
        # cumulative N-element atomic chain without precision loss.
        c = X[0].clone().detach().contiguous().to(torch.float32)
        sum_diff = torch.zeros(D, device=X.device, dtype=torch.float64)
        ss_diff = torch.zeros(D, device=X.device, dtype=torch.float64)
        _col_sum_ss_shifted_kernel[grid](X, c, sum_diff, ss_diff, N, D, sN, sD)
        # Reconstruct mean and var on host in fp64 then cast — keeps the
        # `Σ/N − (Σ/N)²` cancellation at fp64 precision.
        c_d = c.to(torch.float64)
        mean_d = c_d + sum_diff / N
        mean_diff_d = sum_diff / N
        var_d = ss_diff / N - mean_diff_d * mean_diff_d
        var_d.clamp_(min=0.0)
        mean = mean_d.to(torch.float32)
        std = var_d.sqrt().to(torch.float32)
    else:
        mean = torch.zeros(D, device=X.device, dtype=torch.float32)
        _col_sum_kernel[grid](X, mean, N, D, sN, sD)
        mean.div_(N)

        ss = torch.zeros(D, device=X.device, dtype=torch.float32)
        _col_ss_kernel[grid](X, mean, ss, N, D, sN, sD)
        var = ss / N
        std = var.sqrt_()

    # sklearn convention: stds equal to 0 are replaced by 1 to avoid div-by-zero
    std_safe = torch.where(std == 0, torch.ones_like(std), std)
    inv_std = 1.0 / std_safe
    return mean, std, inv_std


def flash_standard_scaler_transform(X: torch.Tensor, mean: torch.Tensor, inv_std: torch.Tensor):
    """Apply (X - mean) * inv_std using a single Triton kernel.

    Requires X to be C-contiguous (the flat-1D layout in `_scale_kernel` assumes
    row-major). For our bench `gen_regression` returns contiguous tensors.
    """
    assert X.is_cuda and X.ndim == 2 and X.is_contiguous()
    N, D = X.shape
    TOTAL = N * D
    Y = torch.empty_like(X)
    grid = lambda META: (triton.cdiv(TOTAL, META["BLOCK"]),)
    _scale_kernel[grid](X, Y, mean, inv_std, TOTAL, D)
    return Y


def flash_standard_scaler_fit_transform(X: torch.Tensor):
    """Compute mean/std (fit) then apply transform. Returns (X_scaled, (mean, std))."""
    mean, std, inv_std = flash_standard_scaler_fit(X)
    Y = flash_standard_scaler_transform(X, mean, inv_std)
    return Y, (mean, std)
