"""Shared helper for the per-batch-size sparse top-k entrypoints.

The sparse path is the fastest top-k route on H100 for IP top-k with D in
(512,768], M a multiple of 64, k<=128. It comes in two ops sharing one
signature (k, num_buckets, buf_cap, select_cap, sample_size):
  * n <= 32  -> hopper_topk.fused_ip_smalln_sparse   (m64n8/m64n32 tile)
  * n  % 64 == 0 (>=64) -> hopper_topk.fused_ip_sparse

Only the sample_size sweet spot differs per batch; buf_cap / select_cap are
both set to M (the whole corpus) so no candidate is ever dropped on overflow.
at::empty over the caching allocator makes the larger buffers free of init cost,
and the select scan is bounded by qcount (not buf_cap), so this costs no latency
while keeping recall maximal. Returns (values=+IP, idx int32).
"""

from __future__ import annotations

import os

import torch

import hopper_ops

_NUM_BUCKETS = int(os.environ.get("HOPPER_NUM_BUCKETS", "64"))


def _caps(m: int, k: int):
    # Both caps = M: the candidate write buffer (buf_cap) and the select-stage
    # lt/eq buffers (select_cap) can hold every gated candidate, eliminating the
    # overflow that dropped recall (e.g. n=8 k=100: 0.9975 -> 1.0). Measured to
    # add no latency (empty alloc + qcount-bounded select scan).
    return m, m


@torch.no_grad()
def sparse_topk_ip(x: torch.Tensor, base: torch.Tensor, k: int, *,
                   sample_size: int, num_buckets: int = _NUM_BUCKETS):
    """Run the fastest sparse IP top-k for the given (already-sized) batch.

    x: [n, D] fp16, base: [M, D] fp16. sample_size is the per-bs tuned value
    (the caller's bsN file supplies it). Picks smalln vs full sparse by n.
    """
    assert x.is_cuda and base.is_cuda, "x/base must be CUDA tensors"
    assert x.dtype == torch.float16 and base.dtype == torch.float16, "fp16 only"
    assert x.dim() == 2 and base.dim() == 2 and x.shape[1] == base.shape[1]
    hopper_ops._ensure_loaded()
    n = x.shape[0]
    m = base.shape[0]
    assert m % 64 == 0, "base rows must be a multiple of 64"
    buf_cap, select_cap = _caps(m, k)
    S = min(m, max(k, sample_size))
    if n <= 32:
        return torch.ops.hopper_topk.fused_ip_smalln_sparse(
            x, base, k, num_buckets, buf_cap, select_cap, S)
    assert n % 64 == 0, "full sparse path requires n to be a multiple of 64"
    return torch.ops.hopper_topk.fused_ip_sparse(
        x, base, k, num_buckets, buf_cap, select_cap, S)
