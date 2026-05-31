"""Ours — single fused kernel (``flash_knn_triton_flashlargek``) large-K KNN.

One matmul + histogram + writeback in a single Triton pass (streaming threshold
+ bucket-partitioned buffer), so the score matrix is never fully materialised.
Times the whole call over ``--iters`` runs (default 100) and reports the mean,
matching the two baselines (``bench_torch_topk.py`` / ``bench_raft_topk.py``).

Run with the python10 env:
    /mnt/hdd/yinziqi/miniconda3/envs/python10/bin/python \\
        efficientlargek/bench_onekernel.py --batch-size 512 --k 10000
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from benchmark import _DEFAULT_BASE, _DEFAULT_QUERY, _time_cuda, read_fvecs
from flashlargek import flash_knn_triton_flashlargek


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

    # ‖c‖² is now computed inside the fused kernel (single corpus pass), so there
    # is no cross-call norm cache to clear — each call reads the corpus once.
    def fn():
        return flash_knn_triton_flashlargek(x, c, k, return_distances=True)

    torch.cuda.reset_peak_memory_stats()
    _time_cuda(
        fn,
        warmup=args.warmup,
        iters=args.iters,
        label=f"flashlargek (one-kernel) B={B} M={M} k={k}",
    )
    print(f"peak GPU mem = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
