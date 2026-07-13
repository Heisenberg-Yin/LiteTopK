"""batch=128 sparse IP top-k entrypoint (fastest route on H100).

Tuned: sample_size = 32768. Full fused_ip_sparse path. With 128 query rows the
per-row base read is well amortized, so the smallest swept sample wins; larger
samples add scan/copy cost that now dominates. Measured ~0.729 ms @ M=1M D=768
k=100.
"""

from __future__ import annotations

import torch

from bs_common import sparse_topk_ip

_SAMPLE = 32768


@torch.no_grad()
def topk_ip_bs128(x: torch.Tensor, base: torch.Tensor, k: int, *, num_buckets: int = 64):
    assert x.shape[0] == 128, "bs128 entrypoint expects batch size 128"
    return sparse_topk_ip(x, base, k, sample_size=_SAMPLE, num_buckets=num_buckets)
