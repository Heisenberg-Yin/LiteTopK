"""Triton tall-skinny GEMM kernels.

Specialised X.T @ X / X.T @ Y / streaming X @ W / X.T @ X covariance
patterns where one or more dims are small (typically D ~ 32–512).
cuBLAS tiles assuming a large output, which under-utilises tensor cores
on these shapes; we tile the *small* output and stream the long dim.

Public surface:
  - triton_cov_gemm(X)        : C = X.T @ X / N (centered/cov form)
  - triton_full_gemm(X)       : C = X.T @ X
  - triton_ab_gemm(A, B)      : C = A.T @ B (tall-skinny)
  - triton_streaming_matmul(X, W) : Y = X @ W (streaming over N)
  - triton_gram_gemm(X)       : Gram matrix X @ X.T
"""

import math


def _round_to_bucket(n):
    """Round n to nearest power-of-2 bucket for autotuner key stability."""
    if n <= 0:
        return 1
    return 1 << math.ceil(math.log2(max(n, 1)))

import torch
import triton
import triton.language as tl


# =============================================================================
# Kernel 1: Tall-Skinny GEMM  —  C = X.T @ X  where X is (N, D), N >> D
#
# cuBLAS handles this poorly because it tiles the output as if it were large,
# but the output is only (D, D). We tile the (D, D) output and stream X in
# panels of BLOCK_N rows, accumulating with tl.dot() for TF32 tensor cores.
# =============================================================================

_TALL_SKINNY_GEMM_CONFIGS = [
    # Focused configs for H100: larger tiles + software pipelining
    triton.Config({"BLOCK_DI": 64, "BLOCK_DJ": 64, "BLOCK_N": 128}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_DI": 64, "BLOCK_DJ": 64, "BLOCK_N": 256}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_DI": 128, "BLOCK_DJ": 64, "BLOCK_N": 128}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_DI": 64, "BLOCK_DJ": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_DI": 128, "BLOCK_DJ": 128, "BLOCK_N": 64}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_DI": 128, "BLOCK_DJ": 128, "BLOCK_N": 128}, num_stages=1, num_warps=8),
    triton.Config({"BLOCK_DI": 32, "BLOCK_DJ": 32, "BLOCK_N": 256}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_DI": 64, "BLOCK_DJ": 64, "BLOCK_N": 64}, num_stages=3, num_warps=4),
]


@triton.autotune(configs=_TALL_SKINNY_GEMM_CONFIGS, key=["N_KEY", "D_KEY"])
@triton.jit
def _tall_skinny_gemm_kernel(
    X_ptr,      # (N, D) input matrix
    OUT_ptr,    # (D, D) output matrix
    N,          # not constexpr — can be very large (2M)
    D,          # not constexpr
    stride_xn,  # stride along N dimension
    stride_xd,  # stride along D dimension
    stride_oi,  # output stride row
    stride_oj,  # output stride col
    N_KEY: tl.constexpr,   # rounded N for autotuner key + loop bound
    D_KEY: tl.constexpr,   # rounded D for autotuner key
    SYMMETRIC: tl.constexpr,  # if True, skip lower-triangle tiles (pid_i > pid_j)
    BLOCK_DI: tl.constexpr,
    BLOCK_DJ: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute C[di, dj] = sum_n X[n, di] * X[n, dj] for a tile of (di, dj)."""
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    di_start = pid_i * BLOCK_DI
    dj_start = pid_j * BLOCK_DJ

    # Skip strictly-lower-triangle tiles for symmetric (X.T @ X). Use
    # row/col coords (not pid coords) because BLOCK_DI may differ from
    # BLOCK_DJ — a tile is entirely below the diagonal iff
    # dj_start + BLOCK_DJ <= di_start.
    if SYMMETRIC:
        if dj_start + BLOCK_DJ <= di_start:
            return

    # D-direction offsets stay int32 (D ≤ ~5000, no overflow risk)
    di_offs = di_start + tl.arange(0, BLOCK_DI)
    dj_offs = dj_start + tl.arange(0, BLOCK_DJ)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_DI, BLOCK_DJ), dtype=tl.float32)

    # Stream X in panels of BLOCK_N rows
    # n_offs needs int64: n * stride_xn can overflow when N * D > 2^31
    for n_start in tl.range(0, N_KEY, BLOCK_N, num_stages=2):
        n_offs = (n_start + tl.arange(0, BLOCK_N)).to(tl.int64)
        n_mask = n_offs < N

        # Load X[:, di_block] -> (BLOCK_N, BLOCK_DI)
        xi_ptrs = X_ptr + n_offs[:, None] * stride_xn + di_offs[None, :] * stride_xd
        xi_mask = n_mask[:, None] & (di_offs[None, :] < D)
        xi = tl.load(xi_ptrs, mask=xi_mask, other=0.0)

        # Load X[:, dj_block] -> (BLOCK_N, BLOCK_DJ)
        xj_ptrs = X_ptr + n_offs[:, None] * stride_xn + dj_offs[None, :] * stride_xd
        xj_mask = n_mask[:, None] & (dj_offs[None, :] < D)
        xj = tl.load(xj_ptrs, mask=xj_mask, other=0.0)

        # Accumulate: acc += xi.T @ xj  (BLOCK_DI x BLOCK_N) @ (BLOCK_N x BLOCK_DJ)
        acc += tl.dot(tl.trans(xi), xj)

    # Store output tile (output is D×D, no overflow risk)
    oi_mask = di_offs[:, None] < D
    oj_mask = dj_offs[None, :] < D
    out_ptrs = OUT_ptr + di_offs[:, None] * stride_oi + dj_offs[None, :] * stride_oj
    tl.store(out_ptrs, acc, mask=oi_mask & oj_mask)


def triton_cov_gemm(X: torch.Tensor) -> torch.Tensor:
    """Compute X.T @ X using Triton tall-skinny GEMM with symmetric optimization.

    Only computes the upper triangle (since X.T@X is symmetric), halving
    the number of tiles. After the kernel, mirrors upper triangle to lower.

    Args:
        X: (N, D) float32 tensor on CUDA, N >> D

    Returns:
        (D, D) float32 tensor = X.T @ X
    """
    assert X.is_cuda and X.ndim == 2
    N, D = X.shape
    X = X.contiguous()

    out = torch.zeros(D, D, device=X.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(D, META["BLOCK_DI"]),
        triton.cdiv(D, META["BLOCK_DJ"]),
    )

    _tall_skinny_gemm_kernel[grid](
        X, out,
        N, D,
        X.stride(0), X.stride(1),
        out.stride(0), out.stride(1),
        N_KEY=_round_to_bucket(N), D_KEY=_round_to_bucket(D),
        SYMMETRIC=True,
    )
    # Mirror upper triangle to lower (negligible cost for D ≤ 5000)
    out = torch.triu(out) + torch.triu(out, diagonal=1).T
    return out


def triton_full_gemm(X: torch.Tensor) -> torch.Tensor:
    """Compute X.T @ X without symmetric optimization (full grid)."""
    assert X.is_cuda and X.ndim == 2
    N, D = X.shape
    X = X.contiguous()

    out = torch.zeros(D, D, device=X.device, dtype=torch.float32)
    grid = lambda META: (
        triton.cdiv(D, META["BLOCK_DI"]),
        triton.cdiv(D, META["BLOCK_DJ"]),
    )
    _tall_skinny_gemm_kernel[grid](
        X, out, N, D,
        X.stride(0), X.stride(1),
        out.stride(0), out.stride(1),
        N_KEY=_round_to_bucket(N), D_KEY=_round_to_bucket(D),
        SYMMETRIC=False,
    )
    return out


# =============================================================================
# Tall-Skinny GEMM variant: C = A.T @ B  where A is (N, P), B is (N, D)
# Used by Truncated SVD: Q.T @ X
# =============================================================================

_AB_GEMM_CONFIGS = [
    triton.Config({"BLOCK_PI": 32, "BLOCK_DJ": 64, "BLOCK_N": 128}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_PI": 32, "BLOCK_DJ": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_PI": 64, "BLOCK_DJ": 64, "BLOCK_N": 128}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_PI": 64, "BLOCK_DJ": 64, "BLOCK_N": 256}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_PI": 64, "BLOCK_DJ": 128, "BLOCK_N": 128}, num_stages=1, num_warps=8),
    triton.Config({"BLOCK_PI": 32, "BLOCK_DJ": 64, "BLOCK_N": 256}, num_stages=2, num_warps=4),
]


@triton.autotune(configs=_AB_GEMM_CONFIGS, key=["N_KEY", "P_KEY", "D_KEY"])
@triton.jit
def _tall_skinny_ab_gemm_kernel(
    A_ptr,      # (N, P) input
    B_ptr,      # (N, D) input
    OUT_ptr,    # (P, D) output
    N, P, D,
    stride_an, stride_ap,
    stride_bn, stride_bd,
    stride_oi, stride_oj,
    N_KEY: tl.constexpr,
    P_KEY: tl.constexpr,
    D_KEY: tl.constexpr,
    BLOCK_PI: tl.constexpr,
    BLOCK_DJ: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute C[pi, dj] = sum_n A[n, pi] * B[n, dj]."""
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    # P and D offsets stay int32 (small dimensions)
    pi_offs = pid_i * BLOCK_PI + tl.arange(0, BLOCK_PI)
    dj_offs = pid_j * BLOCK_DJ + tl.arange(0, BLOCK_DJ)

    acc = tl.zeros((BLOCK_PI, BLOCK_DJ), dtype=tl.float32)

    for n_start in tl.range(0, N_KEY, BLOCK_N, num_stages=2):
        # n_offs int64: n * stride can overflow when N * D > 2^31
        n_offs = (n_start + tl.arange(0, BLOCK_N)).to(tl.int64)
        n_mask = n_offs < N

        a_ptrs = A_ptr + n_offs[:, None] * stride_an + pi_offs[None, :] * stride_ap
        a = tl.load(a_ptrs, mask=n_mask[:, None] & (pi_offs[None, :] < P), other=0.0)

        b_ptrs = B_ptr + n_offs[:, None] * stride_bn + dj_offs[None, :] * stride_bd
        b = tl.load(b_ptrs, mask=n_mask[:, None] & (dj_offs[None, :] < D), other=0.0)

        acc += tl.dot(tl.trans(a), b)

    out_ptrs = OUT_ptr + pi_offs[:, None] * stride_oi + dj_offs[None, :] * stride_oj
    tl.store(out_ptrs, acc, mask=(pi_offs[:, None] < P) & (dj_offs[None, :] < D))


def triton_ab_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compute A.T @ B using Triton tall-skinny GEMM.

    Args:
        A: (N, P) float32 tensor on CUDA
        B: (N, D) float32 tensor on CUDA

    Returns:
        (P, D) float32 tensor = A.T @ B
    """
    assert A.is_cuda and B.is_cuda
    N, P = A.shape
    N2, D = B.shape
    assert N == N2
    A = A.contiguous()
    B = B.contiguous()

    out = torch.zeros(P, D, device=A.device, dtype=torch.float32)
    grid = lambda META: (
        triton.cdiv(P, META["BLOCK_PI"]),
        triton.cdiv(D, META["BLOCK_DJ"]),
    )
    _tall_skinny_ab_gemm_kernel[grid](
        A, B, out,
        N, P, D,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        N_KEY=_round_to_bucket(N), P_KEY=_round_to_bucket(P), D_KEY=_round_to_bucket(D),
    )
    return out


# =============================================================================
# Kernel 1c: Streaming matmul  Y = X @ W  where X is (N, D), W is (D, K), K small
#
# Streams through X rows at full HBM bandwidth while W stays in L2.
# Key property: total HBM = N*D*4 bytes (one pass through X), which at 3.1 TB/s
# gives near-peak BW utilization for large N.
#
# Serves: Randomized PCA/SVD (X @ Omega), CG solver (X @ p)
# =============================================================================

_STREAMING_MATMUL_CONFIGS = [
    triton.Config({"BLOCK_N": 128, "BLOCK_D": 128, "BLOCK_K": 32}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_N": 128, "BLOCK_D": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_N": 256, "BLOCK_D": 64, "BLOCK_K": 32}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_N": 64, "BLOCK_D": 128, "BLOCK_K": 32}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_N": 128, "BLOCK_D": 128, "BLOCK_K": 16}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_N": 256, "BLOCK_D": 128, "BLOCK_K": 16}, num_stages=2, num_warps=8),
]


@triton.autotune(configs=_STREAMING_MATMUL_CONFIGS, key=["N_KEY", "D_KEY", "K_KEY"])
@triton.jit
def _streaming_matmul_kernel(
    X_ptr,      # (N, D)
    W_ptr,      # (D, K)
    OUT_ptr,    # (N, K)
    N, D, K,
    stride_xn, stride_xd,
    stride_wd, stride_wk,
    stride_on, stride_ok,
    N_KEY: tl.constexpr,
    D_KEY: tl.constexpr,
    K_KEY: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Y[block_n, block_k] = X[block_n, :] @ W[:, block_k].

    Each CTA handles a (BLOCK_N, BLOCK_K) tile of the output.
    D is streamed in BLOCK_D chunks with software pipelining.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    n_offs = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = n_offs < N
    k_mask = k_offs < K

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for d_start in tl.range(0, D_KEY, BLOCK_D, num_stages=2):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        # Load X[block_n, block_d] -> (BLOCK_N, BLOCK_D)
        x = tl.load(X_ptr + n_offs[:, None] * stride_xn + d_offs[None, :] * stride_xd,
                     mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        # Load W[block_d, block_k] -> (BLOCK_D, BLOCK_K) — stays in L2
        w = tl.load(W_ptr + d_offs[:, None] * stride_wd + k_offs[None, :] * stride_wk,
                     mask=d_mask[:, None] & k_mask[None, :], other=0.0)

        # TF32 matmul: (BLOCK_N, BLOCK_D) @ (BLOCK_D, BLOCK_K)
        acc += tl.dot(x, w)

    # Store output
    out_ptrs = OUT_ptr + n_offs[:, None] * stride_on + k_offs[None, :] * stride_ok
    tl.store(out_ptrs, acc, mask=n_mask[:, None] & k_mask[None, :])


def triton_streaming_matmul(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Compute Y = X @ W using streaming Triton matmul.

    Optimized for X being tall (N >> D) and W being narrow (K << D).
    Streams through X at full HBM bandwidth while W stays in L2 cache.

    Args:
        X: (N, D) float32 tensor on CUDA
        W: (D, K) float32 tensor on CUDA

    Returns:
        (N, K) float32 tensor
    """
    assert X.is_cuda and W.is_cuda
    N, D = X.shape
    D2, K = W.shape
    assert D == D2
    X = X.contiguous()
    W = W.contiguous()

    out = torch.empty(N, K, device=X.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_N"]),
        triton.cdiv(K, META["BLOCK_K"]),
    )
    _streaming_matmul_kernel[grid](
        X, W, out,
        N, D, K,
        X.stride(0), X.stride(1),
        W.stride(0), W.stride(1),
        out.stride(0), out.stride(1),
        N_KEY=_round_to_bucket(N),
        D_KEY=_round_to_bucket(D),
        K_KEY=_round_to_bucket(K),
    )
    return out


# =============================================================================
# Kernel 4: Gram GEMM  —  G = X @ X.T  where X is (N, D), D >> N
#
# Dual of tall-skinny GEMM: tiles the (N, N) output and streams D in panels.
# Symmetric optimization (X @ X.T is symmetric) halves tile count.
# Used by PCA Lanczos path when D >> N (avoids materializing D×D covariance).
# =============================================================================

_GRAM_GEMM_CONFIGS = [
    triton.Config({"BLOCK_NI": 64, "BLOCK_NJ": 64, "BLOCK_D": 128}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_NI": 64, "BLOCK_NJ": 64, "BLOCK_D": 256}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_NI": 128, "BLOCK_NJ": 64, "BLOCK_D": 128}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_NI": 64, "BLOCK_NJ": 128, "BLOCK_D": 128}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_NI": 128, "BLOCK_NJ": 128, "BLOCK_D": 64}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_NI": 128, "BLOCK_NJ": 128, "BLOCK_D": 128}, num_stages=1, num_warps=8),
    triton.Config({"BLOCK_NI": 32, "BLOCK_NJ": 32, "BLOCK_D": 256}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_NI": 64, "BLOCK_NJ": 64, "BLOCK_D": 64}, num_stages=3, num_warps=4),
]


@triton.autotune(configs=_GRAM_GEMM_CONFIGS, key=["N_KEY", "D_KEY"])
@triton.jit
def _gram_gemm_kernel(
    X_ptr,      # (N, D) input matrix
    OUT_ptr,    # (N, N) output matrix
    N,          # number of samples (small)
    D,          # number of features (large, streamed)
    stride_xn,  # stride along N dimension
    stride_xd,  # stride along D dimension
    stride_oi,  # output stride row
    stride_oj,  # output stride col
    N_KEY: tl.constexpr,
    D_KEY: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    BLOCK_NI: tl.constexpr,
    BLOCK_NJ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute G[ni, nj] = sum_d X[ni, d] * X[nj, d] for a tile of (ni, nj)."""
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    ni_start = pid_i * BLOCK_NI
    nj_start = pid_j * BLOCK_NJ

    # Skip lower triangle for symmetric (X @ X.T) — must use *byte-aligned*
    # row/col coords (not pid coords) because BLOCK_NI may differ from
    # BLOCK_NJ. A tile is entirely in the strict lower triangle iff its
    # smallest column ≥ exclusive end-row of the tile, i.e.
    # nj_start >= ni_start + BLOCK_NI.
    if SYMMETRIC:
        if nj_start + BLOCK_NJ <= ni_start:
            return

    # N-direction offsets — N is small, int32 safe
    ni_offs = ni_start + tl.arange(0, BLOCK_NI)
    nj_offs = nj_start + tl.arange(0, BLOCK_NJ)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_NI, BLOCK_NJ), dtype=tl.float32)

    # Stream D in panels of BLOCK_D columns
    for d_start in tl.range(0, D_KEY, BLOCK_D, num_stages=2):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        # Load X[ni_block, d_block] -> (BLOCK_NI, BLOCK_D)
        xi_ptrs = X_ptr + ni_offs[:, None].to(tl.int64) * stride_xn + d_offs[None, :] * stride_xd
        xi_mask = (ni_offs[:, None] < N) & d_mask[None, :]
        xi = tl.load(xi_ptrs, mask=xi_mask, other=0.0)

        # Load X[nj_block, d_block] -> (BLOCK_NJ, BLOCK_D)
        xj_ptrs = X_ptr + nj_offs[:, None].to(tl.int64) * stride_xn + d_offs[None, :] * stride_xd
        xj_mask = (nj_offs[:, None] < N) & d_mask[None, :]
        xj = tl.load(xj_ptrs, mask=xj_mask, other=0.0)

        # Accumulate: acc += xi @ xj.T  (BLOCK_NI x BLOCK_D) @ (BLOCK_D x BLOCK_NJ)
        acc += tl.dot(xi, tl.trans(xj))

    # Store output tile
    oi_mask = ni_offs[:, None] < N
    oj_mask = nj_offs[None, :] < N
    out_ptrs = OUT_ptr + ni_offs[:, None] * stride_oi + nj_offs[None, :] * stride_oj
    tl.store(out_ptrs, acc, mask=oi_mask & oj_mask)


def triton_gram_gemm(X: torch.Tensor) -> torch.Tensor:
    """Compute X @ X.T using Triton GEMM with symmetric optimization.

    Tiles the small (N, N) output and streams through D (the large dimension).
    Only computes upper triangle, then mirrors. For D >> N, this is the
    dual of triton_cov_gemm.

    Args:
        X: (N, D) float32 tensor on CUDA, D >> N typically

    Returns:
        (N, N) float32 tensor = X @ X.T
    """
    assert X.is_cuda and X.ndim == 2
    N, D = X.shape
    X = X.contiguous()

    out = torch.zeros(N, N, device=X.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_NI"]),
        triton.cdiv(N, META["BLOCK_NJ"]),
    )

    _gram_gemm_kernel[grid](
        X, out,
        N, D,
        X.stride(0), X.stride(1),
        out.stride(0), out.stride(1),
        N_KEY=_round_to_bucket(N), D_KEY=_round_to_bucket(D),
        SYMMETRIC=True,
    )
    # Mirror upper triangle to lower
    out = torch.triu(out) + torch.triu(out, diagonal=1).T
    return out

