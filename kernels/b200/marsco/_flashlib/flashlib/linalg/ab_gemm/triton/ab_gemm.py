"""ab_gemm: A.T @ B for tall-skinny A and B (sharing N dim).

Used by truncated SVD (Q.T @ X projection) and PCA dual path
(X.T @ U_K to recover D-space eigenvectors).
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
    triton.Config({"BLOCK_PI": 32, "BLOCK_DJ": 64, "BLOCK_N": 128}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_PI": 32, "BLOCK_DJ": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_PI": 64, "BLOCK_DJ": 64, "BLOCK_N": 128}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_PI": 64, "BLOCK_DJ": 64, "BLOCK_N": 256}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_PI": 64, "BLOCK_DJ": 128, "BLOCK_N": 128}, num_stages=1, num_warps=8),
    triton.Config({"BLOCK_PI": 32, "BLOCK_DJ": 64, "BLOCK_N": 256}, num_stages=2, num_warps=4),
]


@triton.autotune(configs=_CONFIGS, key=["N_KEY", "P_KEY", "D_KEY"])
@triton.jit
def _tall_skinny_ab_gemm_kernel(
    A_ptr, B_ptr, OUT_ptr,
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
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    pi_offs = pid_i * BLOCK_PI + tl.arange(0, BLOCK_PI)
    dj_offs = pid_j * BLOCK_DJ + tl.arange(0, BLOCK_DJ)

    acc = tl.zeros((BLOCK_PI, BLOCK_DJ), dtype=tl.float32)

    for n_start in tl.range(0, N_KEY, BLOCK_N, num_stages=2):
        n_offs = (n_start + tl.arange(0, BLOCK_N)).to(tl.int64)
        n_mask = n_offs < N

        a_ptrs = A_ptr + n_offs[:, None] * stride_an + pi_offs[None, :] * stride_ap
        a = tl.load(a_ptrs, mask=n_mask[:, None] & (pi_offs[None, :] < P), other=0.0)

        b_ptrs = B_ptr + n_offs[:, None] * stride_bn + dj_offs[None, :] * stride_bd
        b = tl.load(b_ptrs, mask=n_mask[:, None] & (dj_offs[None, :] < D), other=0.0)

        acc += tl.dot(tl.trans(a), b)

    out_ptrs = OUT_ptr + pi_offs[:, None] * stride_oi + dj_offs[None, :] * stride_oj
    tl.store(out_ptrs, acc, mask=(pi_offs[:, None] < P) & (dj_offs[None, :] < D))


def ab_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compute A.T @ B using a tall-skinny GEMM, A: (N, P), B: (N, D).

    Args:
        A: (N, P) float32 CUDA tensor.
        B: (N, D) float32 CUDA tensor.

    Returns:
        (P, D) float32 = A.T @ B.
    """
    assert A.is_cuda and B.is_cuda
    N, P = A.shape
    N2, D = B.shape
    assert N == N2
    A = A.contiguous()
    B = B.contiguous()

    out = torch.zeros(P, D, device=A.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(P, META["BLOCK_PI"]), triton.cdiv(D, META["BLOCK_DJ"]))
    _tall_skinny_ab_gemm_kernel[grid](
        A, B, out, N, P, D,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        N_KEY=_round_to_bucket(N), P_KEY=_round_to_bucket(P), D_KEY=_round_to_bucket(D),
    )
    return out
