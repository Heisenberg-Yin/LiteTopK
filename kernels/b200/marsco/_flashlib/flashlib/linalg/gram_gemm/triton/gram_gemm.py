"""gram_gemm: X @ X.T for D >> N (dual of cov_gemm).

Tiles the small (N, N) output and streams D in panels. Symmetric optimization
halves tile count. Used by PCA when D >> N (avoids materializing D x D cov).
"""
import math
import torch
import triton
import triton.language as tl


def _round_to_bucket(n):
    if n <= 0:
        return 1
    return 1 << math.ceil(math.log2(max(n, 1)))


_CONFIGS = [
    triton.Config({"BLOCK_NI": 64, "BLOCK_NJ": 64, "BLOCK_D": 128}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_NI": 64, "BLOCK_NJ": 64, "BLOCK_D": 256}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_NI": 128, "BLOCK_NJ": 64, "BLOCK_D": 128}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_NI": 64, "BLOCK_NJ": 128, "BLOCK_D": 128}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_NI": 128, "BLOCK_NJ": 128, "BLOCK_D": 64}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_NI": 128, "BLOCK_NJ": 128, "BLOCK_D": 128}, num_stages=1, num_warps=8),
    triton.Config({"BLOCK_NI": 32, "BLOCK_NJ": 32, "BLOCK_D": 256}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_NI": 64, "BLOCK_NJ": 64, "BLOCK_D": 64}, num_stages=3, num_warps=4),
]


@triton.autotune(configs=_CONFIGS, key=["N_KEY", "D_KEY"])
@triton.jit
def _gram_gemm_kernel(
    X_ptr, OUT_ptr,
    N, D,
    stride_xn, stride_xd,
    stride_oi, stride_oj,
    N_KEY: tl.constexpr,
    D_KEY: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    BLOCK_NI: tl.constexpr,
    BLOCK_NJ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    ni_start = pid_i * BLOCK_NI
    nj_start = pid_j * BLOCK_NJ

    if SYMMETRIC:
        if nj_start + BLOCK_NJ <= ni_start:
            return

    ni_offs = ni_start + tl.arange(0, BLOCK_NI)
    nj_offs = nj_start + tl.arange(0, BLOCK_NJ)

    acc = tl.zeros((BLOCK_NI, BLOCK_NJ), dtype=tl.float32)

    for d_start in tl.range(0, D_KEY, BLOCK_D, num_stages=2):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        xi_ptrs = X_ptr + ni_offs[:, None].to(tl.int64) * stride_xn + d_offs[None, :] * stride_xd
        xi = tl.load(xi_ptrs, mask=(ni_offs[:, None] < N) & d_mask[None, :], other=0.0)

        xj_ptrs = X_ptr + nj_offs[:, None].to(tl.int64) * stride_xn + d_offs[None, :] * stride_xd
        xj = tl.load(xj_ptrs, mask=(nj_offs[:, None] < N) & d_mask[None, :], other=0.0)

        acc += tl.dot(xi, tl.trans(xj))

    out_ptrs = OUT_ptr + ni_offs[:, None] * stride_oi + nj_offs[None, :] * stride_oj
    tl.store(out_ptrs, acc, mask=(ni_offs[:, None] < N) & (nj_offs[None, :] < N))


def gram_gemm(X: torch.Tensor, *, tol=None) -> torch.Tensor:
    """X @ X.T (Gram matrix). Used when D >> N as dual of cov_gemm.

    Args:
        X: (N, D) CUDA tensor (any float dtype).
        tol: kept for API compatibility; the inner ``tl.dot`` runs at
            Triton's default precision (TF32 on Hopper for fp32). The
            output is fp32-accumulated.

    Returns:
        (N, N) float32 = X @ X.T.
    """
    del tol
    assert X.is_cuda and X.ndim == 2
    N, D = X.shape
    X = X.contiguous()

    out = torch.zeros(N, N, device=X.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_NI"]), triton.cdiv(N, META["BLOCK_NJ"]))
    _gram_gemm_kernel[grid](
        X, out, N, D,
        X.stride(0), X.stride(1),
        out.stride(0), out.stride(1),
        N_KEY=_round_to_bucket(N), D_KEY=_round_to_bucket(D),
        SYMMETRIC=True,
    )
    out = torch.triu(out) + torch.triu(out, diagonal=1).T
    return out
