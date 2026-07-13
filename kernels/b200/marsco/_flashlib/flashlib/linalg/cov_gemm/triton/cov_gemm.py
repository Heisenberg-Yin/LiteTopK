"""cov_gemm: X.T @ X for tall-skinny X (N >> D).

Tiles the small (D, D) output and streams X in panels of BLOCK_N rows.
Symmetric optimization (X.T @ X is symmetric) halves tile count.

Serves: PCA, TruncSVD, LinReg, Ridge.
"""
import math
import torch
import triton
import triton.language as tl


def _round_to_bucket(n):
    """Round n to nearest power-of-2 bucket for autotune key stability."""
    if n <= 0:
        return 1
    return 1 << math.ceil(math.log2(max(n, 1)))


_CONFIGS = [
    triton.Config({"BLOCK_DI": 64, "BLOCK_DJ": 64, "BLOCK_N": 128}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_DI": 64, "BLOCK_DJ": 64, "BLOCK_N": 256}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_DI": 128, "BLOCK_DJ": 64, "BLOCK_N": 128}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_DI": 64, "BLOCK_DJ": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_DI": 128, "BLOCK_DJ": 128, "BLOCK_N": 64}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_DI": 128, "BLOCK_DJ": 128, "BLOCK_N": 128}, num_stages=1, num_warps=8),
    triton.Config({"BLOCK_DI": 32, "BLOCK_DJ": 32, "BLOCK_N": 256}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_DI": 64, "BLOCK_DJ": 64, "BLOCK_N": 64}, num_stages=3, num_warps=4),
]


@triton.autotune(configs=_CONFIGS, key=["N_KEY", "D_KEY"])
@triton.jit
def _tall_skinny_gemm_kernel(
    X_ptr, OUT_ptr,
    N, D,
    stride_xn, stride_xd,
    stride_oi, stride_oj,
    N_KEY: tl.constexpr,
    D_KEY: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    BLOCK_DI: tl.constexpr,
    BLOCK_DJ: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    di_start = pid_i * BLOCK_DI
    dj_start = pid_j * BLOCK_DJ

    if SYMMETRIC:
        if dj_start + BLOCK_DJ <= di_start:
            return

    di_offs = di_start + tl.arange(0, BLOCK_DI)
    dj_offs = dj_start + tl.arange(0, BLOCK_DJ)

    acc = tl.zeros((BLOCK_DI, BLOCK_DJ), dtype=tl.float32)

    for n_start in tl.range(0, N_KEY, BLOCK_N, num_stages=2):
        n_offs = (n_start + tl.arange(0, BLOCK_N)).to(tl.int64)
        n_mask = n_offs < N

        xi_ptrs = X_ptr + n_offs[:, None] * stride_xn + di_offs[None, :] * stride_xd
        xi = tl.load(xi_ptrs, mask=n_mask[:, None] & (di_offs[None, :] < D), other=0.0)

        xj_ptrs = X_ptr + n_offs[:, None] * stride_xn + dj_offs[None, :] * stride_xd
        xj = tl.load(xj_ptrs, mask=n_mask[:, None] & (dj_offs[None, :] < D), other=0.0)

        acc += tl.dot(tl.trans(xi), xj)

    out_ptrs = OUT_ptr + di_offs[:, None] * stride_oi + dj_offs[None, :] * stride_oj
    tl.store(out_ptrs, acc, mask=(di_offs[:, None] < D) & (dj_offs[None, :] < D))


def cov_gemm(X: torch.Tensor) -> torch.Tensor:
    """Compute X.T @ X using a tall-skinny GEMM with symmetric optimization.

    Only computes upper triangle, then mirrors. For N >> D, this is much faster
    than a generic cuBLAS sgemm because output (D, D) is small enough to stay
    L2-cached while X streams through HBM.

    Args:
        X: (N, D) float32 CUDA tensor, N >> D.

    Returns:
        (D, D) float32 = X.T @ X.
    """
    assert X.is_cuda and X.ndim == 2
    N, D = X.shape
    X = X.contiguous()

    out = torch.zeros(D, D, device=X.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(D, META["BLOCK_DI"]), triton.cdiv(D, META["BLOCK_DJ"]))
    _tall_skinny_gemm_kernel[grid](
        X, out, N, D,
        X.stride(0), X.stride(1),
        out.stride(0), out.stride(1),
        N_KEY=_round_to_bucket(N), D_KEY=_round_to_bucket(D),
        SYMMETRIC=True,
    )
    out = torch.triu(out) + torch.triu(out, diagonal=1).T
    return out


def full_gemm(X: torch.Tensor) -> torch.Tensor:
    """X.T @ X without symmetric optimization (full grid; for testing/comparison)."""
    assert X.is_cuda and X.ndim == 2
    N, D = X.shape
    X = X.contiguous()

    out = torch.zeros(D, D, device=X.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(D, META["BLOCK_DI"]), triton.cdiv(D, META["BLOCK_DJ"]))
    _tall_skinny_gemm_kernel[grid](
        X, out, N, D,
        X.stride(0), X.stride(1),
        out.stride(0), out.stride(1),
        N_KEY=_round_to_bucket(N), D_KEY=_round_to_bucket(D),
        SYMMETRIC=False,
    )
    return out
