"""Shared utilities for flash_knn Triton dispatch + kernels.

Lives in one place so :mod:`dispatch`, :mod:`sortmerge` and :mod:`insert`
all import the same primitives:

  * :func:`_next_pow2` -- D / K padding helper.
  * :func:`_bench_quick` -- micro-bench used by the autotuner.
  * :func:`_fp32_to_sortable_u32` / :func:`_sortable_u32_to_fp32` --
    branch-free IEEE-sortable u32 transform. With the x²-free score
    ``s = c² - 2⟨x, c⟩`` the quantity is signed, so the packed-uint64
    sort-merge / final-sort needs an ascending-u32 = ascending-fp32
    mapping. The transform is 3 ops, exact for non-NaN fp32.
  * :data:`_INF_PACKED` -- (sortable +inf << 32) | -1 idx sentinel,
    used as the initial fill of the sortmerge running top-K so the
    early-exit ``chunk_best < sortable_inv(top)`` always fires until
    real values arrive.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _next_pow2(n: int) -> int:
    """Smallest power-of-two >= ``max(1, n)``. Used for D / K padding."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _bench_quick(fn, *, warmup: int = 3, reps: int = 5) -> float:
    """Median-ish wall time in ms; cheap enough to call once per autotune cfg."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


# Sortable upper-bits = sortable(+inf) = 0xFF800000; lower-bits = -1 (idx
# sentinel) = 0xFFFFFFFF. ``_sortable_u32_to_fp32(0xFF800000) == +inf`` so
# the sortmerge early-exit ``chunk_best < +inf`` always fires until the
# running top-K starts to hold real values.
#
# Wrapped in ``tl.constexpr`` so ``@triton.jit`` kernels can reference it
# directly as a module-level global (un-wrapped Python ints are not
# accessible from inside a JIT'd kernel as of Triton 3.x).
_INF_PACKED = tl.constexpr(0xFF800000_FFFFFFFF)


@triton.jit
def _fp32_to_sortable_u32(x):
    """Map ascending fp32 -> ascending uint32 (branch-free, 3 ops).

    Standard IEEE-754 trick: arithmetic-shift the int32 view by 31 (so
    MSB=1 / negative produces 0xFFFFFFFF, MSB=0 / positive produces 0),
    OR with 0x80000000 to flip the sign bit of positives, then XOR with
    the original bits. NaNs map into 0xFF800001..0xFFFFFFFF (unused in
    practice).
    """
    bits = x.to(tl.uint32, bitcast=True)
    sign_mask = (bits.to(tl.int32, bitcast=True) >> 31).to(tl.uint32, bitcast=True)
    flip = sign_mask | 0x80000000
    return bits ^ flip


@triton.jit
def _sortable_u32_to_fp32(s):
    """Inverse of :func:`_fp32_to_sortable_u32`."""
    sign_mask = (s.to(tl.int32, bitcast=True) >> 31).to(tl.uint32, bitcast=True)
    # In sortable space: MSB=1 -> was positive (flip back via 0x80000000);
    #                    MSB=0 -> was negative (flip back via 0xFFFFFFFF).
    flip = (sign_mask ^ 0xFFFFFFFF) | 0x80000000
    return (s ^ flip).to(tl.float32, bitcast=True)
