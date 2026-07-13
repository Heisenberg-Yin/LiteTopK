"""batch=32 sparse IP top-k entrypoint (fastest route on H100).

Tuned: sample_size = 131072 (same sweet spot as bs8). Still the m64n32 smalln
sparse op. Measured ~0.636 ms @ M=1M D=768 k=100.
"""

from __future__ import annotations

import torch

from bs_common import sparse_topk_ip

_SAMPLE = 131072


@torch.no_grad()
def topk_ip_bs32(x: torch.Tensor, base: torch.Tensor, k: int, *, num_buckets: int = 64):
    assert x.shape[0] == 32, "bs32 entrypoint expects batch size 32"
    return sparse_topk_ip(x, base, k, sample_size=_SAMPLE, num_buckets=num_buckets)
