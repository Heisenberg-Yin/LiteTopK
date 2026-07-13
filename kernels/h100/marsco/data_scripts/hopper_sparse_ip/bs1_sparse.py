"""batch=1 sparse IP top-k entrypoint (fastest route on H100).

Tuned: sample_size = 131072. For n=1 the latency is monotone in S (larger is
slightly faster, S=M is the absolute best at ~0.581 ms) but the curve is flat
past ~131072 — 131072 measures ~0.587 ms, within ~1% of S=M, so we use the same
value as bs8 for consistency. Smaller S (e.g. 2048) can drop recall on some
inputs, so we stay at 131072. Measured @ M=1M D=768 k=100 (torch ~0.66 ms).
"""

from __future__ import annotations

import torch

from bs_common import sparse_topk_ip

_SAMPLE = 131072


@torch.no_grad()
def topk_ip_bs1(x: torch.Tensor, base: torch.Tensor, k: int, *, num_buckets: int = 64):
    assert x.shape[0] == 1, "bs1 entrypoint expects batch size 1"
    return sparse_topk_ip(x, base, k, sample_size=_SAMPLE, num_buckets=num_buckets)
