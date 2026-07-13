"""Triton-only multi-precision matmul — drop-in replacement for nvmath/cuBLAS LT.

Three functions matching the legacy `_nvmath_bf16.py` API exactly, but pure
Triton (no nvmath / cuBLAS LT dependency):

    mm_3xbf16(A_fp32, B_fp32)          fp32 @ fp32 -> fp32 via Ozaki 3xbf16,
                                       single Triton kernel launch.
    mm_tf32_lt(A_fp32, B_fp32)         fp32 @ fp32 -> fp32 via TF32 tensor cores.
    bf16_mm_fp32(A_bf16, B_bf16, C=)   bf16 @ bf16 -> fp32 with fp32 accumulator;
                                       optional C is added if provided (beta=1).

The first two reuse the fused kernels in
`flashlib.linalg.gemm.triton.fused_kernels` which were specifically designed
to beat nvmath by avoiding the four bf16 HBM round-trip stages cuBLAS LT does
(A_hi, A_lo, B_hi, B_lo writes).

The third is a small Triton GEMM that takes bf16 inputs directly — used by
`btrtri` recursion at sub-tile sizes where the Ozaki split would be wasted
work. Standard `tl.dot(a_bf16, b_bf16, acc=acc_fp32)`.
"""
import torch
import triton
import triton.language as tl

from flashlib.linalg.gemm.triton.fused_kernels import (
    fused_ozaki_matmul,
    fused_tf32_matmul,
)


# ──────────────────────────────────────────────────────────────────────────────
# Public API: matches `_nvmath_bf16.py` so call sites can swap import paths.
# ──────────────────────────────────────────────────────────────────────────────


# Cross-over: fused single-launch wins at small N (HBM split cost dominates
# the 3 GEMM compute); 3-launch wins at large N (TC peak utilization
# dominates). Empirical break-even on H200 is N ~ 6000.
_THREE_LAUNCH_MIN_N = 6000


def mm_3xbf16(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """fp32 @ fp32 -> fp32 via Ozaki 3xbf16, ~1e-5 rel err.

    Auto-selects between two paths based on size:

      Fused (single Triton launch):
          Holds A_hi/A_lo/B_hi/B_lo bf16 splits in registers per K-tile,
          one kernel does all 3 dot products. No HBM split staging.
          ~44% TC peak (limited by 2 fp32 accumulators in registers).
          Wins for N < ~6000 where HBM split cost dominates.

      3-launch (split + chain):
          Pre-cast A,B to bf16 in HBM (4 staging buffers), then 3 calls
          to `bf16_mm_fp32` with C-accumulator chaining.
          ~55% TC peak (each kernel runs the dedicated bf16_mm_fp32 with
          only one fp32 accumulator). Wins for N >= ~6000.

    Drop-in replacement for `_nvmath_bf16.mm_3xbf16`.
    """
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    M, K = A.shape
    _, N = B.shape
    if max(M, N, K) < _THREE_LAUNCH_MIN_N:
        return fused_ozaki_matmul(A, B, mode="3xbf16")
    A_hi = A.to(torch.bfloat16)
    A_lo = (A - A_hi.to(torch.float32)).to(torch.bfloat16)
    B_hi = B.to(torch.bfloat16)
    B_lo = (B - B_hi.to(torch.float32)).to(torch.bfloat16)
    C = bf16_mm_fp32(A_hi, B_hi)
    C = bf16_mm_fp32(A_hi, B_lo, C=C)
    C = bf16_mm_fp32(A_lo, B_hi, C=C)
    return C


def mm_3xbf16_fused(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Single-launch fused Ozaki — kept for the small-N regime.

    Saves the bf16 HBM split traffic but caps at ~44% TC peak due to
    two-accumulator register pressure. Faster than `mm_3xbf16` only at
    very small N where the HBM split cost dominates.
    """
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    return fused_ozaki_matmul(A, B, mode="3xbf16")


def mm_tf32_lt(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """fp32 @ fp32 -> fp32 via TF32 tensor cores, single Triton launch.

    Drop-in replacement for `_nvmath_bf16.mm_tf32_lt`. ~1e-3 rel err
    (single-TF32 mode; for 3xTF32 ~1e-7, use mode='tf32x3' on
    fused_tf32_matmul directly).
    """
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    return fused_tf32_matmul(A, B, mode="tf32")


# ──────────────────────────────────────────────────────────────────────────────
# Direct bf16 GEMM with fp32 accumulator (no Ozaki split).
# ──────────────────────────────────────────────────────────────────────────────

_BF16_GEMM_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                  num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                  num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                  num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
                  num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                  num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_M": 8},
                  num_stages=4, num_warps=4),
]


@triton.autotune(configs=_BF16_GEMM_CONFIGS, key=["M", "N", "K", "HAS_C"])
@triton.jit
def _bf16_gemm_fp32_kernel(
    A_ptr, B_ptr, C_ptr, OUT_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_om, stride_on,
    HAS_C: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """bf16 inputs, fp32 accumulation, fp32 output. OUT = A @ B + (C if HAS_C else 0)."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A_block_ptr = A_ptr + rm[:, None].to(tl.int64) * stride_am + rk[None, :] * stride_ak
    B_block_ptr = B_ptr + rk[:, None] * stride_bk + rn[None, :].to(tl.int64) * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = rk[None, :] < (K - k * BLOCK_K)
        a = tl.load(A_block_ptr, mask=(rm[:, None] < M) & k_mask, other=0.0)
        b = tl.load(B_block_ptr, mask=(rk[:, None] < (K - k * BLOCK_K)) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
        A_block_ptr += BLOCK_K * stride_ak
        B_block_ptr += BLOCK_K * stride_bk

    if HAS_C:
        C_block_ptr = C_ptr + rm[:, None].to(tl.int64) * stride_cm + rn[None, :].to(tl.int64) * stride_cn
        c = tl.load(C_block_ptr, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
        acc = acc + c

    OUT_block_ptr = OUT_ptr + rm[:, None].to(tl.int64) * stride_om + rn[None, :].to(tl.int64) * stride_on
    tl.store(OUT_block_ptr, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def bf16_mm_fp32(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor | None = None) -> torch.Tensor:
    """bf16 @ bf16 -> fp32 with fp32 accumulator; optional C added (beta=1).

    Drop-in replacement for `_nvmath_bf16.bf16_mm_fp32`. Uses one Triton kernel
    with `tl.dot(.., acc=acc_fp32)` so accumulation is fp32 throughout.
    """
    assert A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16
    assert A.is_cuda and B.is_cuda and A.ndim == 2 and B.ndim == 2
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    A = A.contiguous()
    B = B.contiguous()
    out = torch.empty(M, N, dtype=torch.float32, device=A.device)

    has_c = C is not None
    if has_c:
        assert C.shape == (M, N) and C.dtype == torch.float32
        C = C.contiguous()
        c_strides = (C.stride(0), C.stride(1))
        C_arg = C
    else:
        c_strides = (0, 0)
        C_arg = A  # any valid fp32-or-bf16 ptr; HAS_C=0 means it's never read

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    _bf16_gemm_fp32_kernel[grid](
        A, B, C_arg, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        c_strides[0], c_strides[1],
        out.stride(0), out.stride(1),
        HAS_C=1 if has_c else 0,
    )
    return out


def clear_plan_cache():
    """No-op (no plan cache in the pure-Triton path). Kept for API parity."""
    pass


__all__ = [
    "mm_3xbf16", "mm_tf32_lt", "bf16_mm_fp32",
    "clear_plan_cache",
]
