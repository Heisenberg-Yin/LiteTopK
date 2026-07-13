"""Triton helpers around the CuTe DSL GEMM — fused split + fused sum."""
import torch
import triton
import triton.language as tl


@triton.jit
def _split_fp32_to_bf16_kernel(
    X_ptr, HI_ptr, LO_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    hi = x.to(tl.bfloat16)
    lo = (x - hi.to(tl.float32)).to(tl.bfloat16)
    tl.store(HI_ptr + offs, hi, mask=mask)
    tl.store(LO_ptr + offs, lo, mask=mask)


def split_fp32_to_bf16_fused(X: torch.Tensor):
    """One-pass split: returns (X_hi, X_lo) both bf16, contiguous.

    Vs torch: 0.3 ms vs 1.5 ms at N=8192×8192.
    """
    assert X.dtype == torch.float32 and X.is_cuda and X.is_contiguous()
    N = X.numel()
    X_hi = torch.empty_like(X, dtype=torch.bfloat16)
    X_lo = torch.empty_like(X, dtype=torch.bfloat16)
    BLOCK = 4096
    grid = (triton.cdiv(N, BLOCK),)
    _split_fp32_to_bf16_kernel[grid](X, X_hi, X_lo, N, BLOCK=BLOCK, num_warps=4)
    return X_hi.view_as(X), X_lo.view_as(X)


@triton.jit
def _sum3_kernel(
    A_ptr, B_ptr, C_ptr, OUT_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = tl.load(C_ptr + offs, mask=mask, other=0.0)
    tl.store(OUT_ptr + offs, a + b + c, mask=mask)


def sum3_fused(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
               out: torch.Tensor | None = None):
    """out = A + B + C in a single HBM pass. All fp32, same shape."""
    assert A.shape == B.shape == C.shape
    if out is None:
        out = torch.empty_like(A)
    N = A.numel()
    BLOCK = 4096
    grid = (triton.cdiv(N, BLOCK),)
    _sum3_kernel[grid](A.contiguous(), B.contiguous(), C.contiguous(),
                       out, N, BLOCK=BLOCK, num_warps=4)
    return out
