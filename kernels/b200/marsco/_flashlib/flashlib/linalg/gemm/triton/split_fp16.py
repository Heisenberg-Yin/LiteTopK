"""FP32/FP64 → 2× FP16 split helpers.

FP16 has 10-bit mantissa (vs BF16's 7-bit), so a 2-component FP16 split
captures ~20 mantissa bits (vs BF16x3's ~14 bits) at the same wgmma
throughput. Cost: FP16's 5-bit exponent caps |x| ≤ 65504, so this only
works for inputs in moderate dynamic range (typical ML + small-range
scientific). Inputs outside this range need scaling per tile — not done
here.

For inputs that DO fit, this is a Pareto-better drop-in replacement for
bf16x3.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _split_fp32_to_fp16_kernel(x_ptr, hi_ptr, lo_ptr, n,
                                BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # Clamp to FP16 representable range to avoid Inf/NaN.
    fp16_max = 65504.0
    x_clamped = tl.minimum(tl.maximum(x, -fp16_max), fp16_max)
    hi = x_clamped.to(tl.float16)
    # The residual lives at relative magnitude 2^-10 of x, so it fits
    # comfortably in fp16 if x itself fit.
    lo_fp32 = x_clamped - hi.to(tl.float32)
    lo_fp32_clamped = tl.minimum(tl.maximum(lo_fp32, -fp16_max), fp16_max)
    lo = lo_fp32_clamped.to(tl.float16)
    tl.store(hi_ptr + offs, hi, mask=mask)
    tl.store(lo_ptr + offs, lo, mask=mask)


def split_fp32_fp16_pair(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 → (fp16_hi, fp16_lo). Combined ~20 mantissa bits.

    Note: caller must ensure |x| ≤ 65504 (FP16 max). Out-of-range values
    are clamped (Pareto choice — most ML/typical scientific data fits).
    """
    assert x.dtype == torch.float32 and x.is_cuda
    hi = torch.empty_like(x, dtype=torch.float16)
    lo = torch.empty_like(x, dtype=torch.float16)
    n = x.numel()
    BLOCK = 4096
    grid = (triton.cdiv(n, BLOCK),)
    _split_fp32_to_fp16_kernel[grid](x.contiguous().view(-1), hi.view(-1),
                                       lo.view(-1), n, BLOCK=BLOCK, num_warps=4)
    return hi, lo


@triton.jit
def _split_fp64_to_fp16_kernel(x_ptr, hi_ptr, lo_ptr, n,
                                BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)             # fp64
    fp16_max = 65504.0
    x_clamped = tl.minimum(tl.maximum(x, -fp16_max), fp16_max)
    hi = x_clamped.to(tl.float16)
    lo_fp32 = (x_clamped - hi.to(tl.float64)).to(tl.float32)
    lo_fp32_c = tl.minimum(tl.maximum(lo_fp32, -fp16_max), fp16_max)
    lo = lo_fp32_c.to(tl.float16)
    tl.store(hi_ptr + offs, hi, mask=mask)
    tl.store(lo_ptr + offs, lo, mask=mask)


def split_fp64_fp16_pair(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 → (fp16_hi, fp16_lo). Combined ~20 mantissa bits with FP16 range."""
    assert x.dtype == torch.float64 and x.is_cuda
    hi = torch.empty_like(x, dtype=torch.float16)
    lo = torch.empty_like(x, dtype=torch.float16)
    n = x.numel()
    BLOCK = 4096
    grid = (triton.cdiv(n, BLOCK),)
    _split_fp64_to_fp16_kernel[grid](x.contiguous().view(-1), hi.view(-1),
                                       lo.view(-1), n, BLOCK=BLOCK, num_warps=4)
    return hi, lo
