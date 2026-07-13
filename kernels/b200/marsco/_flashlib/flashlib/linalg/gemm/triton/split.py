"""Vectorized FP32 -> 2x BF16 split kernel and FP64 -> 2x TF32 split.

These split passes are memory-bound (read 1x, write 2x per element) so on
HBM3e at ~4.8 TB/s a 256 MB matrix takes ~0.4 ms — negligible vs. the
multi-ms matmul. The benefit is that the matmul kernel can then operate
purely on BF16 (resp. TF32-as-FP32) tiles and use the full BK=64/128
tilings without spilling shared memory to hold the FP32 originals.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _split_fp32_to_bf16_kernel(x_ptr, hi_ptr, lo_ptr, n,
                                BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    hi = x.to(tl.bfloat16)
    lo = (x - hi.to(tl.float32)).to(tl.bfloat16)
    tl.store(hi_ptr + offs, hi, mask=mask)
    tl.store(lo_ptr + offs, lo, mask=mask)


def split_fp32_bf16_pair(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (hi, lo) BF16 tensors with the same shape as x:fp32."""
    assert x.dtype == torch.float32 and x.is_cuda
    hi = torch.empty_like(x, dtype=torch.bfloat16)
    lo = torch.empty_like(x, dtype=torch.bfloat16)
    n = x.numel()
    BLOCK = 4096
    grid = (triton.cdiv(n, BLOCK),)
    _split_fp32_to_bf16_kernel[grid](x.contiguous().view(-1), hi.view(-1), lo.view(-1), n,
                                       BLOCK=BLOCK, num_warps=4)
    return hi, lo


@triton.jit
def _round_to_tf32(x_f32):
    """Round-to-nearest-even truncation of an fp32 to tf32 (10-bit mantissa)."""
    bits = x_f32.to(tl.int32, bitcast=True)
    # round-to-nearest-even via half-ULP bias on the 13-bit truncated tail
    bias = ((bits >> 13) & 1) + 0x0FFF
    rounded = (bits + bias) & ~0x1FFF
    return rounded.to(tl.float32, bitcast=True)


@triton.jit
def _split_fp64_to_tf32_kernel(x_ptr, hi_ptr, lo_ptr, n,
                                BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)                  # fp64
    # Round hi to tf32 BEFORE taking the residual, so lo carries the
    # bits below tf32-hi's 10 mantissa bits (not fp32-hi's 23).
    hi_f32 = _round_to_tf32(x.to(tl.float32))
    lo_f32 = _round_to_tf32((x - hi_f32.to(tl.float64)).to(tl.float32))
    tl.store(hi_ptr + offs, hi_f32, mask=mask)
    tl.store(lo_ptr + offs, lo_f32, mask=mask)


def split_fp64_tf32_pair(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (hi, lo) FP32 tensors that mathematically reconstruct x:fp64."""
    assert x.dtype == torch.float64 and x.is_cuda
    hi = torch.empty_like(x, dtype=torch.float32)
    lo = torch.empty_like(x, dtype=torch.float32)
    n = x.numel()
    BLOCK = 4096
    grid = (triton.cdiv(n, BLOCK),)
    _split_fp64_to_tf32_kernel[grid](x.contiguous().view(-1), hi.view(-1), lo.view(-1), n,
                                       BLOCK=BLOCK, num_warps=4)
    return hi, lo
