"""Fast row sum-of-squares (||x_i||^2) helper + per-tensor cache.

Tiny utility, but shared by the FA3 fused KNN path which needs the
pre-computed query/corpus norms to fold into the squared-L2 distance
formula ``||x - c||^2 = ||x||^2 + ||c||^2 - 2 <x, c>``.

This module deliberately stays narrow:
  - ``_row_sq_kernel`` is a pure Triton row-norm kernel (no cross
    matrix, no top-K) — it never materialises an N x M tensor.
  - ``_fast_row_sq`` wraps the kernel.
  - ``_get_or_compute_csq`` caches results per ``(data_ptr, shape,
    dtype)`` so repeat queries on the same corpus skip the recompute.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from flashlib.primitives.knn.triton._common import _next_pow2


@triton.jit
def _row_sq_kernel(
    x_ptr, out_ptr,
    stride_x_b, stride_x_n, stride_x_d,
    stride_o_b, stride_o_n,
    B: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BN: tl.constexpr, BD: tl.constexpr,
):
    """Row sum-of-squares for ``(B, N, D) -> (B, N)`` fp32.

    Casts input to fp32 inside the kernel; reads input exactly once
    and writes output exactly once. At BN=128 BD=128 nw=4 this hits
    ~36% peak HBM BW on H200 (typical for tall-skinny reads).
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)
    n_offs = (pid_n * BN + tl.arange(0, BN)).to(tl.int64)
    n_mask = n_offs < N
    acc = tl.zeros([BN], dtype=tl.float32)
    for d_start in range(0, D, BD):
        d_offs = (d_start + tl.arange(0, BD)).to(tl.int64)
        d_mask = d_offs < D
        x = tl.load(
            x_ptr + pid_b * stride_x_b
            + n_offs[:, None] * stride_x_n
            + d_offs[None, :] * stride_x_d,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        )
        x_f = x.to(tl.float32)
        acc += tl.sum(x_f * x_f, axis=1)
    tl.store(out_ptr + pid_b * stride_o_b + n_offs * stride_o_n,
             acc, mask=n_mask)


def _fast_row_sq(x: torch.Tensor) -> torch.Tensor:
    """Return ``(B, N)`` fp32 row sum-of-squares via the Triton kernel."""
    assert x.is_cuda and x.ndim == 3
    B, N, D = x.shape
    out = torch.empty(B, N, device=x.device, dtype=torch.float32)
    BN = 128 if N >= 128 else _next_pow2(max(N, 16))
    BD = 128 if D >= 128 else _next_pow2(max(D, 16))
    grid = (math.ceil(N / BN), B)
    _row_sq_kernel[grid](
        x, out,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1),
        B=B, N=N, D=D, BN=BN, BD=BD,
        num_warps=4,
    )
    return out


# Per-data-ptr cache so repeated queries on the same corpus don't pay
# the row-norm kernel cost.
_csq_cache: dict = {}


def _get_or_compute_csq(c: torch.Tensor) -> torch.Tensor:
    """Cached ``||c_i||^2`` (fp32) keyed by ``(data_ptr, shape, dtype)``."""
    key = (c.data_ptr(), tuple(c.shape), c.dtype)
    cached = _csq_cache.get(key)
    if cached is not None and cached.device == c.device:
        return cached
    csq = _fast_row_sq(c)
    _csq_cache[key] = csq
    if len(_csq_cache) > 8:
        for old_key in list(_csq_cache.keys())[:-8]:
            del _csq_cache[old_key]
    return csq
