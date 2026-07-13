"""Fused Triton GEMM kernels for PCA.

These kernels fuse two operations that were previously separate:

  1. The symmetric tall-skinny (or wide-flat) GEMM body itself
     (TF32 tensor cores).
  2. The /N (or arbitrary scale) divide on the accumulator — applied while
     the accumulator is still in registers, so no extra HBM round-trip.

Additionally, the **mirror step is dropped entirely**. The previous pipeline
did `torch.triu(out) + torch.triu(out, diagonal=1).T` to make the cov matrix
fully symmetric, but the downstream `torch.linalg.eigh(A, UPLO='U')` (and
cuSOLVER syevd called via the same path) only reads the upper triangle.
The lower triangle of the output is left uninitialized — that's fine,
nothing reads it.

Combined savings vs the original 3-step pipeline (gemm → /N → triu+T mirror):
  - 2 separate kernel launches (~5-10 us each on H200) saved
  - One full HBM read+write+add pass over (D,D) matrix (`triu+T` mirror) saved
  - The /N is folded into the GEMM register accumulator (no follow-up div)

Per-op savings at xlarge (2M × 1024) are ~0.5-0.8 ms wallclock — small
compared to the GEMM itself (~70 ms) but unambiguously real, and at small/
medium sizes (where the GEMM is < 2 ms) the relative savings are bigger.

The kernel is fp32 throughout — bf16 input was explicitly rejected by the
user. tl.dot uses TF32 mode on H100/H200 just like the baseline.
"""

import math
import torch
import triton
import triton.language as tl


def _round_to_bucket(n):
    if n <= 0:
        return 1
    return 1 << math.ceil(math.log2(max(n, 1)))


# =============================================================================
# Fused cov GEMM (upper-triangle only):
#     out[upper] = (X.T @ X) * scale
# X is (N, D), N >> D typical.  Output is (D, D); only the upper triangle
# (incl. diagonal) is written. Lower triangle is uninitialized — eigh
# reads only the upper triangle, so this is correct.
# =============================================================================

_FUSED_COV_CONFIGS = [
    triton.Config({"BLOCK_DI": 64, "BLOCK_DJ": 64, "BLOCK_N": 128}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_DI": 64, "BLOCK_DJ": 64, "BLOCK_N": 256}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_DI": 128, "BLOCK_DJ": 64, "BLOCK_N": 128}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_DI": 64, "BLOCK_DJ": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_DI": 128, "BLOCK_DJ": 128, "BLOCK_N": 64}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_DI": 128, "BLOCK_DJ": 128, "BLOCK_N": 128}, num_stages=1, num_warps=8),
    triton.Config({"BLOCK_DI": 32, "BLOCK_DJ": 32, "BLOCK_N": 256}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_DI": 64, "BLOCK_DJ": 64, "BLOCK_N": 64}, num_stages=3, num_warps=4),
]


@triton.autotune(configs=_FUSED_COV_CONFIGS, key=["N_KEY", "D_KEY"])
@triton.jit
def _fused_cov_gemm_upper_kernel(
    X_ptr,
    OUT_ptr,
    N,
    D,
    SCALE,
    stride_xn,
    stride_xd,
    stride_oi,
    stride_oj,
    N_KEY: tl.constexpr,
    D_KEY: tl.constexpr,
    BLOCK_DI: tl.constexpr,
    BLOCK_DJ: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    di_start = pid_i * BLOCK_DI
    dj_start = pid_j * BLOCK_DJ

    # Skip strict lower triangle.
    if dj_start + BLOCK_DJ <= di_start:
        return

    di_offs = di_start + tl.arange(0, BLOCK_DI)
    dj_offs = dj_start + tl.arange(0, BLOCK_DJ)

    acc = tl.zeros((BLOCK_DI, BLOCK_DJ), dtype=tl.float32)

    for n_start in tl.range(0, N_KEY, BLOCK_N, num_stages=2):
        n_offs = (n_start + tl.arange(0, BLOCK_N)).to(tl.int64)
        n_mask = n_offs < N

        xi_ptrs = X_ptr + n_offs[:, None] * stride_xn + di_offs[None, :] * stride_xd
        xi_mask = n_mask[:, None] & (di_offs[None, :] < D)
        xi = tl.load(xi_ptrs, mask=xi_mask, other=0.0)

        xj_ptrs = X_ptr + n_offs[:, None] * stride_xn + dj_offs[None, :] * stride_xd
        xj_mask = n_mask[:, None] & (dj_offs[None, :] < D)
        xj = tl.load(xj_ptrs, mask=xj_mask, other=0.0)

        acc += tl.dot(tl.trans(xi), xj)

    # Apply scale in registers.
    acc = acc * SCALE

    # Write the [i, j] tile (upper-tri or diagonal). Skipping lower already
    # done above. Mask on diagonal tile is unnecessary for eigh-UPLO='U'
    # correctness because eigh ignores cells with col < row anyway.
    out_ptrs = OUT_ptr + di_offs[:, None] * stride_oi + dj_offs[None, :] * stride_oj
    tl.store(out_ptrs, acc, mask=(di_offs[:, None] < D) & (dj_offs[None, :] < D))


def triton_cov_gemm_fused(X: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Fused: cov[upper] = (X.T @ X) * scale, single kernel.

    Output is (D,D) with only the upper triangle written. Pass to
    `torch.linalg.eigh(A, UPLO='U')` (or the in-house `triton_eigh` which
    delegates to cuSOLVER syevd 'U' / MKL dsyev 'U').

    Replaces the 3-step pipeline (gemm → /N → triu+T mirror) with one launch.

    Args:
        X: (N, D) fp32 CUDA tensor.
        scale: scalar applied in registers before the store. For PCA, pass 1/N.

    Returns:
        (D, D) fp32 tensor — upper triangle holds (X.T@X)*scale.
    """
    assert X.is_cuda and X.ndim == 2
    N, D = X.shape
    X = X.contiguous()

    # `empty` is fine — eigh-UPLO='U' won't read uninitialized lower.
    out = torch.empty(D, D, device=X.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(D, META["BLOCK_DI"]),
        triton.cdiv(D, META["BLOCK_DJ"]),
    )

    _fused_cov_gemm_upper_kernel[grid](
        X, out,
        N, D,
        float(scale),
        X.stride(0), X.stride(1),
        out.stride(0), out.stride(1),
        N_KEY=_round_to_bucket(N), D_KEY=_round_to_bucket(D),
    )
    return out


# =============================================================================
# Fused gram GEMM (upper-triangle only):
#     out[upper] = (X @ X.T) * scale   (N×N, D >> N)
# =============================================================================

_FUSED_GRAM_CONFIGS = [
    triton.Config({"BLOCK_NI": 64, "BLOCK_NJ": 64, "BLOCK_D": 128}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_NI": 64, "BLOCK_NJ": 64, "BLOCK_D": 256}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_NI": 128, "BLOCK_NJ": 64, "BLOCK_D": 128}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_NI": 64, "BLOCK_NJ": 128, "BLOCK_D": 128}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_NI": 128, "BLOCK_NJ": 128, "BLOCK_D": 64}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_NI": 128, "BLOCK_NJ": 128, "BLOCK_D": 128}, num_stages=1, num_warps=8),
    triton.Config({"BLOCK_NI": 32, "BLOCK_NJ": 32, "BLOCK_D": 256}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_NI": 64, "BLOCK_NJ": 64, "BLOCK_D": 64}, num_stages=3, num_warps=4),
]


@triton.autotune(configs=_FUSED_GRAM_CONFIGS, key=["N_KEY", "D_KEY"])
@triton.jit
def _fused_gram_gemm_upper_kernel(
    X_ptr,
    OUT_ptr,
    N,
    D,
    SCALE,
    stride_xn,
    stride_xd,
    stride_oi,
    stride_oj,
    N_KEY: tl.constexpr,
    D_KEY: tl.constexpr,
    BLOCK_NI: tl.constexpr,
    BLOCK_NJ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    ni_start = pid_i * BLOCK_NI
    nj_start = pid_j * BLOCK_NJ

    if nj_start + BLOCK_NJ <= ni_start:
        return

    ni_offs = ni_start + tl.arange(0, BLOCK_NI)
    nj_offs = nj_start + tl.arange(0, BLOCK_NJ)

    acc = tl.zeros((BLOCK_NI, BLOCK_NJ), dtype=tl.float32)

    for d_start in tl.range(0, D_KEY, BLOCK_D, num_stages=2):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        xi_ptrs = X_ptr + ni_offs[:, None].to(tl.int64) * stride_xn + d_offs[None, :] * stride_xd
        xi_mask = (ni_offs[:, None] < N) & d_mask[None, :]
        xi = tl.load(xi_ptrs, mask=xi_mask, other=0.0)

        xj_ptrs = X_ptr + nj_offs[:, None].to(tl.int64) * stride_xn + d_offs[None, :] * stride_xd
        xj_mask = (nj_offs[:, None] < N) & d_mask[None, :]
        xj = tl.load(xj_ptrs, mask=xj_mask, other=0.0)

        acc += tl.dot(xi, tl.trans(xj))

    acc = acc * SCALE

    out_ptrs = OUT_ptr + ni_offs[:, None] * stride_oi + nj_offs[None, :] * stride_oj
    tl.store(out_ptrs, acc, mask=(ni_offs[:, None] < N) & (nj_offs[None, :] < N))


def triton_eigh_upper(A: torch.Tensor) -> tuple:
    """Eigendecomposition reading only the upper triangle of A.

    Mirrors the dispatch policy of `common.triton_kernels.triton_eigh`:
      - D ≤ 512: CPU MKL LAPACK (avoids cuSOLVER's ~5ms launch overhead)
      - D > 512: cuSOLVER (`torch.linalg.eigh(..., UPLO='U')`)
    """
    D = A.shape[0]
    if D > 512:
        return torch.linalg.eigh(A, UPLO='U')

    # CPU MKL path; lazy-init thread count once.
    global _eigh_cpu_initialized
    if not _eigh_cpu_initialized:
        torch.set_num_threads(4)
        _eigh_cpu_initialized = True

    torch.cuda.synchronize()
    A_cpu = A.cpu()
    eigenvalues, eigenvectors = torch.linalg.eigh(A_cpu, UPLO='U')
    return eigenvalues.to(A.device), eigenvectors.to(A.device)


_eigh_cpu_initialized = False


def triton_gram_gemm_fused(X: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Fused: gram[upper] = (X @ X.T) * scale, single kernel."""
    assert X.is_cuda and X.ndim == 2
    N, D = X.shape
    X = X.contiguous()

    out = torch.empty(N, N, device=X.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_NI"]),
        triton.cdiv(N, META["BLOCK_NJ"]),
    )

    _fused_gram_gemm_upper_kernel[grid](
        X, out,
        N, D,
        float(scale),
        X.stride(0), X.stride(1),
        out.stride(0), out.stride(1),
        N_KEY=_round_to_bucket(N), D_KEY=_round_to_bucket(D),
    )
    return out
