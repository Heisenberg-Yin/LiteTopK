"""batch=64 sparse IP top-k entrypoint (fastest route on H100).

Tuned: sample_size = 65536. n=64 is a multiple of 64, so it uses the full
fused_ip_sparse path (not the smalln tile). With more query rows the optimal
sample shrinks further. Measured ~0.649 ms @ M=1M D=768 k=100.
"""

from __future__ import annotations

import torch

from bs_common import sparse_topk_ip

_SAMPLE = 65536


@torch.no_grad()
def topk_ip_bs64(x: torch.Tensor, base: torch.Tensor, k: int, *, num_buckets: int = 64):
    assert x.shape[0] == 64, "bs64 entrypoint expects batch size 64"
    return sparse_topk_ip(x, base, k, sample_size=_SAMPLE, num_buckets=num_buckets)
