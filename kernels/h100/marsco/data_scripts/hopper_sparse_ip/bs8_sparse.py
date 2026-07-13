"""batch=8 sparse IP top-k entrypoint (fastest route on H100).

Tuned: sample_size = 131072. With 8 query rows the base read is amortized, so a
moderate sample tightens the gate (fewer tail candidates) without the large-S
overhead that hurts n=1. Measured ~0.596 ms @ M=1M D=768 k=100.
"""

from __future__ import annotations

import torch

from bs_common import sparse_topk_ip

_SAMPLE = 131072


@torch.no_grad()
def topk_ip_bs8(x: torch.Tensor, base: torch.Tensor, k: int, *, num_buckets: int = 64):
    assert x.shape[0] == 8, "bs8 entrypoint expects batch size 8"
    return sparse_topk_ip(x, base, k, sample_size=_SAMPLE, num_buckets=num_buckets)
