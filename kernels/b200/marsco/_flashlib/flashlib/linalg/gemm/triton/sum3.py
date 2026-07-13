"""Fused 3-tensor element-wise sum (fp32) for the cuBLAS TF32x6 path.

Used by :mod:`flashlib.linalg.gemm.native.cublas_tf32x6` to collapse
the three back-to-back FP32-with-TF32 cuBLAS GEMM outputs into a
single accumulator without materialising an intermediate
``c1 + c2`` tensor. Single load-three / store-one pass; ~3 fp32
read + 1 fp32 write per element.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sum3_fp32_kernel(c1_ptr, c2_ptr, c3_ptr, out_ptr, n,
                       BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(c1_ptr + offs, mask=mask)
    b = tl.load(c2_ptr + offs, mask=mask)
    c = tl.load(c3_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a + b + c, mask=mask)


def fused_sum3(c1: torch.Tensor, c2: torch.Tensor,
                c3: torch.Tensor, out: torch.Tensor) -> None:
    """One-pass sum of 3 fp32 tensors -> fp32 ``out`` (read 3, write 1).

    All four tensors must be the same numel; reshapes them to 1-D
    internally so the caller can pass arbitrary contiguous tensors.
    """
    n = c1.numel()
    BLOCK = 4096
    grid = (triton.cdiv(n, BLOCK),)
    _sum3_fp32_kernel[grid](
        c1.view(-1), c2.view(-1), c3.view(-1), out.view(-1), n,
        BLOCK=BLOCK, num_warps=4,
    )
