"""Disabled benchmark entry point.

Use ``benchmark.py`` as the single active benchmark script. The previous
baseline implementation is commented below for reference.
"""

'''
"""Baseline 1 — ``torch.matmul`` + ``torch.topk`` large-K KNN.

For a batch of ``--batch-size`` queries against the corpus, computes the
squared-L2 score ``s = ‖c‖² − 2⟨x,c⟩`` (same argmin-K as ‖x−c‖²) with a single
matmul, then ``torch.topk`` smallest-k. Times the whole call over ``--iters``
runs (default 100) and reports the mean.

Run with the python10 env:
    /mnt/hdd/yinziqi/miniconda3/envs/python10/bin/python \\
        efficientlargek/bench_torch_topk.py --batch-size 512 --k 10000
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from benchmark import _DEFAULT_BASE, _DEFAULT_QUERY, _time_cuda, read_fvecs


@torch.no_grad()
def torch_matmul_topk(x, c, k):
    """``(B,D)``×``(M,D)`` → top-k smallest squared-L2 per query.

    ``csq = ‖c‖²`` is computed here (inside the timed region) so the
    corpus-side preprocessing is counted, matching the other two baselines.
    ``vals`` are the *true* squared-L2 distances ``‖x−c‖² = s + ‖x‖²``,
    recovered from the shifted score ``s = ‖c‖² − 2⟨x,c⟩`` (same as flashlargek)."""
    csq = (c * c).sum(dim=-1)  # ‖c‖²
    cross = torch.matmul(x, c.t())  # (B, M)
    score = csq.unsqueeze(0) - 2.0 * cross  # (B, M), argmin == argmin ‖x−c‖²
    vals, idxs = torch.topk(score, k, dim=-1, largest=False, sorted=False)
    xsq = (x * x).sum(dim=-1)  # ‖x‖²
    vals = (vals + xsq.unsqueeze(1)).clamp_min_(0.0)  # true ‖x−c‖²
    return vals, idxs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=_DEFAULT_BASE)
    p.add_argument("--query", default=_DEFAULT_QUERY)
    p.add_argument("--num-base", type=int, default=1_000_000, help="corpus rows M")
    p.add_argument("--batch-size", type=int, default=512, help="queries per call")
    p.add_argument("--k", type=int, default=10_000)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    assert torch.cuda.is_available(), "needs a CUDA GPU"
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    X = torch.from_numpy(read_fvecs(args.base, limit=args.num_base)).to(device).float()
    Q = torch.from_numpy(read_fvecs(args.query)).to(device).float()
    M, D = X.shape
    B = args.batch_size
    k = min(args.k, M)
    x = Q[torch.arange(B, device=device) % Q.shape[0]].contiguous()  # tile to fill bs
    c = X  # corpus
    assert x.shape[1] == D, f"query dim {x.shape[1]} != corpus dim {D}"
    print(f"M={M} D={D} batch_size={B} (of {Q.shape[0]} queries) k={k}")

    fn = lambda: torch_matmul_topk(x, c, k)

    torch.cuda.reset_peak_memory_stats()
    _time_cuda(
        fn,
        warmup=args.warmup,
        iters=args.iters,
        label=f"torch.matmul+torch.topk B={B} M={M} k={k}",
    )
    print(f"peak GPU mem = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
'''
