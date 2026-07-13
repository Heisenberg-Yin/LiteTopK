"""Fused Triton kernels for flash linear regression.

Three fusions, listed by impact at xlarge (N=2M, D=5K):

1. ``cast_and_xty`` — fuse fp32→bf16 cast of X with the fp32 Xty = X.T @ y GEMV.
   Both reads X. Single pass replaces two passes.
   Saves ~9ms at xlarge (one full fp32 read of X = 40GB / 4500GB/s).

2. ``xtr_from_bf16`` — recompute Xtr using bf16 X (already-materialized) instead
   of fp32 X. Reads half the bytes; iter-refinement still works because the
   bf16 rounding error in Xtr is below the chol_solve back-substitution error.
   Saves ~3ms on the second refine GEMV at xlarge.

3. ``residual_from_bf16`` — same trick for the residual r = y - X @ w; uses
   bf16 X (already in memory). Saves ~3ms at xlarge. Slightly less accurate
   than fp32 residual but still within the rel ≤ 2e-3 tolerance after one
   cholesky-solve correction step.

Combined: cast→xty saves 1 X-pass; refine reuses bf16 X (no fresh fp32 X reads).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Fused cast (fp32→bf16) + Xty (X.T @ y in fp32)
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _cast_xty_kernel(
    X_fp32_ptr,
    X_bf16_ptr,
    y_ptr,
    Xty_ptr,        # (D,) fp32, atomic-add target
    N, D,
    stride_xn, stride_xd,
    stride_xb_n, stride_xb_d,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    N_LOOPS: tl.constexpr,
):
    """One block handles BLOCK_N * N_LOOPS rows × BLOCK_D cols of X.

    Sequential inner loop along N reduces atomic_add congestion (one atomic
    per BLOCK_D tile per program — partial accumulated across N_LOOPS rows
    in registers first).

    For each (BLOCK_N, BLOCK_D) sub-tile:
      - Reads X[n_tile, d_tile] (fp32)
      - Writes X_bf16[n_tile, d_tile] (bf16)
      - Reads y[n_tile]
      - Accumulates X.T[d_tile, n_tile] @ y[n_tile] into register
    Final: atomic-adds the whole BLOCK_D accumulator into Xty[d_tile].
    """
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_d64 = offs_d.to(tl.int64)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    n_base = pid_n * BLOCK_N * N_LOOPS
    for li in range(N_LOOPS):
        offs_n = n_base + li * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_n64 = offs_n.to(tl.int64)
        mask = mask_n[:, None] & mask_d[None, :]
        x_ptrs  = X_fp32_ptr + offs_n64[:, None] * stride_xn  + offs_d64[None, :] * stride_xd
        xb_ptrs = X_bf16_ptr + offs_n64[:, None] * stride_xb_n + offs_d64[None, :] * stride_xb_d

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        tl.store(xb_ptrs, x.to(tl.bfloat16), mask=mask)
        yv = tl.load(y_ptr + offs_n, mask=mask_n, other=0.0)
        acc += tl.sum(x * yv[:, None], axis=0)

    tl.atomic_add(Xty_ptr + offs_d, acc, mask=mask_d)


def cast_and_xty(X: torch.Tensor, y: torch.Tensor):
    """Fused fp32→bf16 cast of X and fp32 Xty = X.T @ y. Returns (X_bf16, Xty)."""
    assert X.is_contiguous() and X.dtype == torch.float32
    assert y.is_contiguous() and y.dtype == torch.float32
    N, D = X.shape

    X_bf = torch.empty((N, D), dtype=torch.bfloat16, device=X.device)
    Xty = torch.zeros(D, dtype=torch.float32, device=X.device)

    # Tuned for H200 SXM at xlarge (N=2M, D=5K): BN=64, BD=512, NL=4
    # gives ~19.7 ms vs torch's 23.9 ms (cast 13.7 + xty 10.0).
    BLOCK_N = 64
    BLOCK_D = 512 if D >= 512 else 256 if D >= 256 else 128 if D >= 128 else 64
    N_LOOPS = 4
    grid = (triton.cdiv(N, BLOCK_N * N_LOOPS), triton.cdiv(D, BLOCK_D))
    _cast_xty_kernel[grid](
        X, X_bf, y, Xty,
        N, D,
        X.stride(0), X.stride(1),
        X_bf.stride(0), X_bf.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, N_LOOPS=N_LOOPS,
        num_warps=8,
    )
    return X_bf, Xty


# ──────────────────────────────────────────────────────────────────────────────
# Refine pass 1: residual r = y - X @ w using bf16 X
# (X is already materialized in bf16 from the cast step — reuse it.)
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _residual_bf16_kernel(
    Xb_ptr, w_ptr, y_ptr, r_ptr,
    N, D,
    stride_xn, stride_xd,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """r[n] = y[n] - sum_d X_bf16[n, d] * w[d]  (sum in fp32)"""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_n64 = offs_n.to(tl.int64)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        offs_d64 = offs_d.to(tl.int64)
        x_ptrs = Xb_ptr + offs_n64[:, None] * stride_xn + offs_d64[None, :] * stride_xd
        x = tl.load(x_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + offs_d, mask=mask_d, other=0.0)
        acc += tl.sum(x * wv[None, :], axis=1)

    y = tl.load(y_ptr + offs_n, mask=mask_n, other=0.0)
    tl.store(r_ptr + offs_n, y - acc, mask=mask_n)


# ──────────────────────────────────────────────────────────────────────────────
# Refine pass 2: Xtr = X.T @ r using bf16 X
# Split-N with register accumulation across N_LOOPS to amortize atomic adds.
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _xtr_bf16_kernel(
    Xb_ptr, r_ptr, Xtr_ptr,
    N, D,
    stride_xn, stride_xd,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    N_LOOPS: tl.constexpr,
):
    """Xtr[d] = sum_n X_bf16[n, d] * r[n] (fp32 accumulator)."""
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_d64 = offs_d.to(tl.int64)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    n_base = pid_n * BLOCK_N * N_LOOPS
    for li in range(N_LOOPS):
        offs_n = n_base + li * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_n64 = offs_n.to(tl.int64)
        x_ptrs = Xb_ptr + offs_n64[:, None] * stride_xn + offs_d64[None, :] * stride_xd
        x = tl.load(x_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        r = tl.load(r_ptr + offs_n, mask=mask_n, other=0.0)
        acc += tl.sum(x * r[:, None], axis=0)
    tl.atomic_add(Xtr_ptr + offs_d, acc, mask=mask_d)


def fused_refine_from_bf16(X_bf: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
    """One full refine sweep using already-materialized bf16 X.

    Computes Xtr = X.T @ (y - X @ w), reading X_bf only (no fp32 X needed).
    """
    assert X_bf.dtype == torch.bfloat16 and X_bf.is_contiguous()
    assert y.dtype == torch.float32 and w.dtype == torch.float32
    N, D = X_bf.shape

    # ---- pass 1: residual ----
    r = torch.empty(N, dtype=torch.float32, device=X_bf.device)
    BN1, BD1 = 64, 256
    grid_r = (triton.cdiv(N, BN1),)
    _residual_bf16_kernel[grid_r](
        X_bf, w, y, r,
        N, D,
        X_bf.stride(0), X_bf.stride(1),
        BLOCK_N=BN1, BLOCK_D=BD1,
        num_warps=4,
    )

    # ---- pass 2: Xtr ----
    Xtr = torch.zeros(D, dtype=torch.float32, device=X_bf.device)
    BN2, BD2, NL = 64, 256, 16
    grid_x = (triton.cdiv(N, BN2 * NL), triton.cdiv(D, BD2))
    _xtr_bf16_kernel[grid_x](
        X_bf, r, Xtr,
        N, D,
        X_bf.stride(0), X_bf.stride(1),
        BLOCK_N=BN2, BLOCK_D=BD2, N_LOOPS=NL,
        num_warps=8,
    )
    return Xtr
