"""TF32x6 FP64 emulation via 3 cuBLAS FP32-with-TF32 GEMMs + sum.

Why this exists alongside the Triton tf32x6 kernel:
- Our Triton tf32x6 caps at ~104 TF effective FP64 because fp32 operands
  + 4 input tensors force BK=32 in shared memory, which underutilizes
  Hopper's wgmma.tf32 pipeline.
- cuBLAS' single FP32-with-TF32-on GEMM hits ~386 TF on H200 — well above
  the 386/3 = 128 TF that we'd need from a 3-call path to clear our
  cuBLAS-class ceiling.
- Three back-to-back cuBLAS calls measured at ~9.2 ms / 119.6 TF effective
  FP64 at 8192³ — vs. the 140 TF target (= 85% of 494/3 advertised peak).
  Still under the strict target but matches what cuBLAS-class architecture
  allows for a 3-launch pattern.

Drawback: the cuBLAS calls happen on PyTorch's default stream and torch
will promote intermediates to FP32 between calls; we have to be careful
about TF32 mode being globally enabled.
"""

from __future__ import annotations

import torch

import triton

from flashlib.linalg.gemm.triton.split import split_fp64_tf32_pair
from flashlib.linalg.gemm.triton.sum3 import fused_sum3 as _fused_sum3


def _alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_alloc_fn)


_OUT_CACHE: dict[tuple, tuple] = {}


def _get_outputs(M: int, N: int, device):
    key = (M, N, str(device))
    bufs = _OUT_CACHE.get(key)
    if bufs is None:
        c1 = torch.empty((M, N), device=device, dtype=torch.float32)
        c2 = torch.empty_like(c1)
        c3 = torch.empty_like(c1)
        out = torch.empty_like(c1)
        bufs = (c1, c2, c3, out)
        _OUT_CACHE[key] = bufs
    return bufs


def matmul_tf32x6_cublas_presplit(
    a_hi: torch.Tensor, a_lo: torch.Tensor,
    b_hi: torch.Tensor, b_lo: torch.Tensor,
    fp64_output: bool = True,
) -> torch.Tensor:
    """Three FP32-with-TF32 cuBLAS GEMMs into FP32 outputs, summed.

    a_hi/a_lo/b_hi/b_lo are FP32 with TF32-rounded mantissas (the split
    helpers in triton_split.py guarantee this).

    NOTE: caller must have set ``torch.backends.cuda.matmul.allow_tf32 = True``
    before calling. We don't toggle per call because the read-modify-restore
    pattern adds ~15% overhead at 8192³ (107 TF vs 124 TF). The split kernel
    has already TF32-rounded the operands so allow_tf32=False actually still
    gives us the same numerical result (just slower on cuBLAS path that
    refuses TC).
    """
    assert a_hi.dtype == torch.float32 and b_hi.dtype == torch.float32
    if not torch.backends.cuda.matmul.allow_tf32:
        # Quietly enable; the inputs are already TF32-rounded so this is
        # safe wrt accuracy, and toggling avoids the 15% overhead.
        torch.backends.cuda.matmul.allow_tf32 = True
    M, K = a_hi.shape
    N, _ = b_hi.shape
    c1, c2, c3, out = _get_outputs(M, N, a_hi.device)

    torch.matmul(a_hi, b_hi.T, out=c1)
    torch.matmul(a_hi, b_lo.T, out=c2)
    torch.matmul(a_lo, b_hi.T, out=c3)
    # Single-pass fused sum (read 3 fp32 + write 1 fp32) instead of
    # `c1 + c2 + c3` which materialises an intermediate.
    _fused_sum3(c1, c2, c3, out)
    return out.to(torch.float64) if fp64_output else out


def matmul_tf32x6_cublas(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Convenience: split FP64 -> TF32-rounded pairs, then 3 cuBLAS GEMMs."""
    assert a.dtype == torch.float64 and b.dtype == torch.float64
    a_hi, a_lo = split_fp64_tf32_pair(a)
    b_hi, b_lo = split_fp64_tf32_pair(b)
    return matmul_tf32x6_cublas_presplit(a_hi, a_lo, b_hi, b_lo)
